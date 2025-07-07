import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from models.clip import clip
from models.cocoop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder

from models.SEM import Semantic_Enhancement_Module

from sklearn.cluster import DBSCAN
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch

def integrate_masks(pred):
    batch_size, num_masks, size, _ = pred.shape
    integrated_masks = torch.zeros((batch_size, 1, size, size), dtype=pred.dtype, device=pred.device)
    for b in range(batch_size):
        for i in range(size):
            for j in range(size):
                pixel_values = pred[b, :, i, j]
                if torch.any(pixel_values > 0):
                    integrated_masks[b, 0, i, j] = torch.max(pixel_values)
    return integrated_masks


def count_non_zero_pixels(target):
    target=target.cpu()
    image_array = np.array(target)
    print(np.sum(image_array > 0))
    return np.sum(image_array > 0)

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Net(nn.Module):
    def __init__(self, args, input_dim, out_dim, dino_pretrained='dinov2_vitb14'):
        super().__init__()
        self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        # set up a vision embedder
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_pretrained).cuda()

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ')for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts              
        self.aff_text_encoder = TextEncoder(clip_model)

        self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)

        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)

        self.linear_cls = nn.Linear(input_dim, out_dim)

        # ------------------ 初始化SEM ----------------------------
        self.sem = Semantic_Enhancement_Module(num_class=self.num_aff, groups=15)
        # --------------------------------------------------------

        self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])


        self.step=0

    def get_orb_mask(self, image_tensor, keypoints):
        original_device = image_tensor.device
        b, _, h, w = image_tensor.shape
        masks = []
        gaussian = self.gaussian_kernel().to(original_device)
        pad = 4  # 9//2

        for i in range(b):
            mask = torch.zeros(h, w, device=original_device)
            kp_coords = keypoints[i]  # (N, 2)
            
            for x, y in kp_coords:
                x, y = int(x), int(y)
                if x-pad < 0 or y-pad < 0 or x+pad >= w or y+pad >= h:
                    continue
                current_patch = mask[y-pad:y+pad+1, x-pad:x+pad+1]
                mask[y-pad:y+pad+1, x-pad:x+pad+1] = torch.maximum(current_patch, gaussian).to(original_device)

            masks.append(torch.clip(mask, 0, 2))
        
        output_tensor = torch.stack(masks).unsqueeze(1).to(original_device)
        return output_tensor
    
    def gaussian_kernel(self, size=9, sigma=1.0):
        ax = torch.linspace(-(size // 2), size // 2, size)
        # xx, yy = torch.meshgrid(ax, ax)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel
    
    def save_pic(self,target):
        target[target>0]=255
        image=Image.fromarray(target.cpu().numpy(),mode='L')
        path="./pic/"+str(self.step)+".png"
        self.step+=1
        image.save(path)
        print("图片已经保存")
        print("图片路径：{}".format(path))


    def forward(self, img, keypoints=None, label=None, gt_aff=None):
        b, _, h, w = img.shape

        # DINO特征提取
        dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        merge_weight = torch.softmax(self.merge_weight, dim=0)
        
        dino_dense = 0
        for i, feat in enumerate(dino_out):
            feat_ = self.lln_linear[i](feat[0])
            feat_ = self.lln_norm[i](feat_)
            dino_dense += feat_ * merge_weight[i]
        dino_dense = self.lln_norm_1(self.embedder(dino_dense))

        # -----------------sem---------------------
        dino_dense = self.sem(dino_dense, label)     
        # -----------------------------------------

        # prompts = self.prompt_learner()
        # ----------------cocoop-------------------------
        prompts0 = self.prompt_learner(dino_dense)
        prompts = prompts0.squeeze(0)
        # -----------------------------------------------
        tokenized_prompts = self.tokenized_prompts
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))

        dino_cls = dino_out[-1][1]
        dino_cls = self.linear_cls(dino_cls)

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)  
        text_features, attn_out, _ = self.seg_decoder(text_features, dino_dense, dino_cls)
        
        attn = (text_features[-1] @ dino_dense.transpose(-2,-1)) * (512**-0.5)
        attn_out = torch.sigmoid(attn)
        attn_out = attn_out.reshape(b, -1, h // 14, w // 14)
        pred = F.interpolate(attn_out, img.shape[-2:], mode='bilinear', align_corners=False)


        if keypoints is not None:
            orb_mask = self.get_orb_mask(img, keypoints)
            orb_mask = F.interpolate(orb_mask, img.shape[-2:], mode='bilinear', align_corners=False)

            scaled_orb_mask = 1 + (orb_mask - orb_mask.min()) / (orb_mask.max() - orb_mask.min() + 1e-8)
            pred = scaled_orb_mask * pred
            pred = torch.clamp(pred, 0, 1)

        
        if self.training:
            assert not label == None, 'Label should be provided during training'
            loss_bce = nn.BCELoss()(pred, label / 255.0)
            loss_dict = {'bce': loss_bce}
            return pred, loss_dict

        else:
            if gt_aff is not None:
                out = torch.zeros(b, h, w).cuda()
                for b_ in range(b):
                    out[b_] = pred[b_, gt_aff[b_]]
                return out

    def _freeze_stages(self, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False
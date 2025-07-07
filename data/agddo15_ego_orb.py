import os
from os.path import join as opj
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

import cv2

BASE_OBJ = ['bag', 'cloth_hat', 'cloth_sleeveless', 'curtains', 'dress', 'hat', 'long_T-shirt', 
            'mask', 'pants', 'short_T-shirt', 'shorts', 'socks', 'tie', 'tissue', 'towel']
SEEN_AFF = ['clip', 'grasp_hat', 'grasp_pantswaist', 'grasp_shoulder', 'grasp_sleeve', 'grasp_strap',
            'hang', 'pick',  'place', 'pull', 'pull_out', 'put_center', 'put_hem', 'put_pantslegs', 'roll']
UNSEEN_AFF = []

class TrainData(data.Dataset):
    def __init__(self, data_root, divide, resize_size=256, crop_size=224):

        self.data_root = data_root
        divide_path = 'one-shot-' + divide.lower()
        self.train_path = opj(data_root, divide_path)

        self.img_ann_list = []
        for file in os.listdir(self.train_path):
            if file.endswith('.jpg'):
                self.img_ann_list.append([file, file.replace('jpg', 'npy')])


        num_obj = 15 if divide == 'Seen' else 7
        assert len(self.img_ann_list) == num_obj, "Each object should only provide one sample"

        self.resize_size = resize_size
        self.crop_size = crop_size

        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)

    def extract_orb_keypoints(self, image_tensor):
        image_np = image_tensor.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype(np.uint8)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        kp = self.orb.detect(gray, None)

        kp_coords = torch.tensor([p.pt for p in kp], dtype=torch.float32)  # (N, 2)
        return kp_coords

    def __getitem__(self, item):
        img, ann = self.img_ann_list[item]
        img = Image.open(opj(self.train_path, img)).convert('RGB')
        ann = torch.from_numpy(np.load(opj(self.train_path, ann)))
        img, ann = self.transform(img, ann)

        keypoints = self.extract_orb_keypoints(img)

        
        return img, ann, keypoints

    def transform(self, img, mask):
        resize = transforms.Resize(size=(self.resize_size, self.resize_size), antialias=None)
        img, mask = resize(img), resize(mask)

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return img, mask

    def __len__(self):
        return len(self.img_ann_list)


class TestData(data.Dataset):
    def __init__(self, data_root, divide, crop_size=224):

        self.data_root = opj(data_root, divide, 'testset')
        self.ego_path = opj(self.data_root, 'egocentric')
        self.mask_path = opj(self.data_root, 'GT')
        self.divide = divide

        self.image_list = []
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
        
        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)

        files = os.listdir(self.ego_path)
        for file in files:
            file_path = os.path.join(self.ego_path, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    mask_path = os.path.join(self.mask_path, file, obj_file, img[:-3] + "png")

                    if os.path.exists(mask_path):
                        self.image_list.append(img_path)

    def extract_orb_keypoints(self, image_tensor):
        image_np = image_tensor.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype(np.uint8)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        kp = self.orb.detect(gray, None)

        kp_coords = torch.tensor([p.pt for p in kp], dtype=torch.float32)  # (N, 2)
        return kp_coords

    def __getitem__(self, item):

        image_path = self.image_list[item]
        names = image_path.split("/")
        aff_name, object = names[-3], names[-2]

        image = self.load_img(image_path)
        AFF_CLASS = SEEN_AFF if self.divide == 'Seen' else UNSEEN_AFF
        gt_aff = AFF_CLASS.index(aff_name)
        names = image_path.split("/")
        mask_path = os.path.join(self.mask_path, names[-3], names[-2], names[-1][:-3] + "png")

        keypoints = self.extract_orb_keypoints(image)

        return image, gt_aff, object, mask_path, keypoints

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):

        return len(self.image_list)
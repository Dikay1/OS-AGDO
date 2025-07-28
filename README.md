# One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes
## Abstract
  Deformable object manipulation in robotics presents significant challenges due to uncertainties in component properties, diverse configurations, visual interference, and ambiguous prompts. These factors complicate both perception and control tasks. To address these challenges, we propose a novel method for One-Shot Affordance Grounding of Deformable Objects (OS-AGDO) in egocentric organizing scenes, enabling robots to recognize previously unseen deformable objects with varying colors and shapes using minimal samples. Specifically, we first introduce the Deformable Object Semantic Enhancement Module (DefoSEM), which enhances hierarchical understanding of the internal structure and improves the ability to accurately identify local features, even under conditions of weak component information. Next, we propose the ORB-Enhanced Keypoint Fusion Module (OEKFM), which optimizes feature extraction of key components by leveraging geometric constraints and improves adaptability to diversity and visual interference. Additionally, we propose an instance-conditional prompt based on image data and task context, effectively mitigates the issue of region ambiguity caused by prompt words. To validate these methods, we construct a diverse real-world dataset, AGDDO15, which includes 15 common types of deformable objects and their associated organizational actions. Experimental results demonstrate that our approach significantly outperforms state-of-the-art methods, achieving improvements of 6.2%, 3.2%, and 2.9% in KLD, SIM, and NSS metrics, respectively, while exhibiting high generalization performance.
 
  ![image](https://github.com/Dikay1/OS-AGDO/blob/main/assets/frame.jpg)
  ###### Fig.1. Illustration of OS-AGDO, the proposed one-shot affordance grounding framework. Our designs are highlighted in four color blocks, which are the visual and text encoders, the CLS-guided transformer decoder, the DefoSEM module, and the Geometric Constraints module. [CLS] denotes the CLS token of the vision encoder.


## TODO

- [x] Release AGDDO15 dataset.
- [x] Release the code.
- [x] Release the [arxiv preprint](https://arxiv.org/pdf/2503.01092).

## Citation
If our work is helpful to you, please consider citing us by using the following BibTeX entry:

```
@article{jia2025one,
  title={One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes},
  author={Jia, Wanjun and Yang, Fan and Duan, Mengfei and Chen, Xianchi and Wang, Yinxi and Jiang, Yiming and Chen, Wenrui and Yang, Kailun and Li, Zhiyong},
  journal={arXiv preprint arXiv:2503.01092},
  year={2025}
}
```

## Usage
### 1.Requirements
  Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.3 
  
```
pip install -r requirements.txt
```
### 2.Dataset
  Download the AGDDO15 dataset from [Baidu Pan]( https://pan.baidu.com/s/1KV4PrwBExB8A5MDq9ZxDgw?pwd=S7U2)[S7U2].(you can annotate your own one-shot data in the same format).
  
  Put the data in the `dataset` folder with the following structure:  
```
dataset 
├── one-shot-seen
└── Seen
```
### 3.Train and Test
  Run following commands to start training or testing:

```
python train.py
python test.py --model_file <PATH_TO_MODEL>
```
 

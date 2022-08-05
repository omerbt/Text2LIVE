# Text2LIVE: Text-Driven Layered Image and Video Editing (ECCV 2022 - Oral)
## [<a href="https://text2live.github.io/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-Text2LIVE-b31b1b.svg)](https://arxiv.org/abs/2204.02491)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/weizmannscience/text2live)

![teaser](https://user-images.githubusercontent.com/22198039/179798581-ca6f6652-600a-400a-b21b-713fc5c15d56.png)

**Text2LIVE** is a method for text-driven editing of real-world images and videos, as described in <a href="https://arxiv.org/abs/2204.02491" target="_blank">(link to paper)</a>.

[//]: # (. It can be used for localized and global edits that change the texture of existing objects or augment the scene with semi-transparent effects &#40;e.g. smoke, fire, snow&#41;.)

[//]: # (### Abstract)
>We present a method for zero-shot, text-driven appearance manipulation in natural images and videos. Specifically, given an input image or video and a target text prompt, our goal is to edit the appearance of existing objects (e.g., object's texture) or augment the scene with new visual effects (e.g., smoke, fire) in a semantically meaningful manner. Our framework trains a generator using an internal dataset of training examples, extracted from a single input (image or video and target text prompt), while leveraging an external pre-trained CLIP model to establish our losses. Rather than directly generating the edited output, our key idea is to generate an edit layer (color+opacity) that is composited over the original input. This allows us to constrain the generation process and maintain high fidelity to the original input via novel text-driven losses that are applied directly to the edit layer. Our method neither relies on a pre-trained generator nor requires user-provided edit masks. Thus, it can perform localized, semantic edits on high-resolution natural images and videos across a variety of objects and scenes.


## Getting Started
### Installation

```
git clone https://github.com/omerbt/Text2LIVE.git
conda create --name text2live python=3.9 
conda activate text2live 
pip install -r requirements.txt
```

### Download sample images and videos
Download sample images and videos from the DAVIS dataset:
```
cd Text2LIVE
gdown https://drive.google.com/uc?id=1osN4PlPkY9uk6pFqJZo8lhJUjTIpa80J&export=download
unzip data.zip
```
It will create a folder `data`:
```
Text2LIVE
├── ...
├── data
│   ├── pretrained_nla_models # NLA models are stored here
│   ├── images # sample images
│   └── videos # sample videos from DAVIS dataset
│         ├── car-turn # contains video frames 
│         ├── ...
└── ...
```
To enforce temporal consistency in video edits, we utilize the Neural Layered Atlases (NLA). Pretrained NLA models are taken from <a href="https://layered-neural-atlases.github.io">here</a>, and are already inside the `data` folder.

### Run examples 
* Our method is designed to change textures of existing objects / augment the scene with semi-transparent effects (e.g., smoke, fire). It is not designed for adding new objects or significantly deviating from the original spatial layout.
* Training **Text2LIVE** multiple times with the same inputs can lead to slightly different results.
* CLIP sometimes exhibits bias towards specific solutions (see figure 9 in the paper), thus slightly different text prompts may lead to different flavors of edits.


The required GPU memory depends on the input image/video size, but you should be good with a Tesla V100 32GB :).
Currently mixed precision introduces some instability in the training process, but it could be added later.

#### Video Editing
Run the following command to start training
```
python train_video.py --example_config car-turn_winter.yaml
```
#### Image Editing
Run the following command to start training
```
python train_image.py --example_config golden_horse.yaml
```
Intermediate results will be saved to `results` during optimization. The frequency of saving intermediate results is indicated in the `log_images_freq` flag of the configuration.

## Sample Results
https://user-images.githubusercontent.com/22198039/179797381-983e0453-2e5d-40e8-983d-578217b358e4.mov

For more see the [supplementary material](https://text2live.github.io/sm/index.html).


## Citation
```
@article{bar2022text2live,
         title     = {Text2LIVE: Text-Driven Layered Image and Video Editing},
         author    = {Bar-Tal, Omer and Ofri-Amar, Dolev and Fridman, Rafail and Kasten, Yoni and Dekel, Tali},
         journal   = {arXiv preprint arXiv:2204.02491},
         year      = {2022}
}
```

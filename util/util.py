from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from madgrad import MADGRAD
from torchvision import transforms


def get_optimizer(cfg, params):
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    elif cfg["optimizer"] == "radam":
        optimizer = torch.optim.RAdam(params, lr=cfg["lr"])
    elif cfg["optimizer"] == "madgrad":
        optimizer = MADGRAD(params, lr=cfg["lr"], weight_decay=0.01, momentum=0.9)
    elif cfg["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg["lr"], weight_decay=0.01)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg["lr"])
    else:
        return NotImplementedError("optimizer [%s] is not implemented", cfg["optimizer"])
    return optimizer


def get_text_criterion(cfg):
    if cfg["text_criterion"] == "spherical":
        text_criterion = spherical_dist_loss
    elif cfg["text_criterion"] == "cosine":
        text_criterion = cosine_loss
    else:
        return NotImplementedError("text criterion [%s] is not implemented", cfg["text_criterion"])
    return text_criterion


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return ((x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()


def cosine_loss(x, y, scaling=1.2):
    return scaling * (1 - F.cosine_similarity(x, y).mean())


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_screen_template():
    return [
        "{} over a green screen.",
        "{} in front of a green screen.",
    ]


def get_augmentations_template():
    templates = [
        "photo of {}.",
        "high quality photo of {}.",
        "a photo of {}.",
        "the photo of {}.",
        "image of {}.",
        "an image of {}.",
        "high quality image of {}.",
        "a high quality image of {}.",
        "the {}.",
        "a {}.",
        "{}.",
        "{}",
        "{}!",
        "{}...",
    ]
    return templates


def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]


def get_mask_boundary(img, mask):
    mask = mask.squeeze()  # mask.shape -> (H, W)
    if torch.sum(mask) > 0:
        y, x = torch.where(mask)
        y0, x0 = y.min(), x.min()
        y1, x1 = y.max(), x.max()
        return img[:, :, y0:y1, x0:x1]
    else:
        return img


def load_video(folder: str, resize=(432, 768), num_frames=70):
    resy, resx = resize
    folder = Path(folder)
    input_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))[:num_frames]
    video = torch.zeros((len(input_files), 3, resy, resx))

    for i, file in enumerate(input_files):
        video[i] = transforms.ToTensor()(Image.open(str(file)).resize((resx, resy), Image.LANCZOS))

    return video

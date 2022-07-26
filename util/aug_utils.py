import warnings
from collections.abc import Sequence
import numbers
import torchvision.transforms as T
from torchvision.transforms.functional import (
    InterpolationMode,
    _interpolation_modes_from_int,
    get_image_num_channels,
    get_image_size,
    perspective,
    crop,
)
import torch
import numpy as np


class RandomScale(object):
    def __init__(self, scale_range=(0.8, 1.2), min_size=None):
        super(RandomScale, self).__init__()
        self.scale_range = scale_range
        self.min_size = min_size if min_size is not None else 0

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size[-2:]
        s = np.random.uniform(*self.scale_range)
        resize_h = max(int(height * s), self.min_size)
        resize_w = max(int(width * s), self.min_size)
        size = (resize_h, resize_w)
        return T.Resize(size)(img)


class RandomSizeCrop(object):
    def __init__(self, min_cover):
        super(RandomSizeCrop, self).__init__()
        self.min_cover = min_cover

    def __call__(self, img):
        if self.min_cover == 1:
            return img
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size[-2:]
        s = np.random.uniform(self.min_cover, 1)
        size_h = int(h * s)
        size_w = int(w * s)
        return T.RandomCrop((size_h, size_w))(img)


class DivisibleCrop(object):
    def __init__(self, d):
        super(DivisibleCrop, self).__init__()
        self.d = d

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size[-2:]

        h = h - h % self.d
        w = w - w % self.d
        return T.CenterCrop((h, w))(img)


class ToTensorSafe(object):
    def __init__(self):
        super(ToTensorSafe, self).__init__()

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img
        return T.ToTensor()(img)


class BorderlessRandomPerspective(object):
    """Applies random perspective and crops the image to be without borders

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        self.p = p

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_crop_endpoints(endpoints):
        topleft, topright, botright, botleft = endpoints
        topy = max(topleft[1], topright[1])
        leftx = max(topleft[0], botleft[0])
        boty = min(botleft[1], botright[1])
        rightx = min(topright[0], botright[0])

        h = boty - topy
        w = rightx - leftx
        return topy, leftx, h, w

    def __call__(self, img):
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            width, height = get_image_size(img)
            startpoints, endpoints = T.RandomPerspective.get_params(width, height, self.distortion_scale)
            warped = perspective(img, startpoints, endpoints, self.interpolation, fill)
            i, j, h, w = self.get_crop_endpoints(endpoints)
            # print(f"Crop size: {h,w}")
            cropped = crop(warped, i, j, h, w)
            return T.Compose([T.Resize(224), T.CenterCrop(224)])(cropped)
        return img

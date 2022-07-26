import random

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.image_model import Model


class VideoModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.net_preprocess = transforms.Compose([])

    @staticmethod
    def resize_crops(crops, resize_factor):
        return torchvision.transforms.functional.resize(
            crops,
            [
                crops.shape[-2] // resize_factor,
                crops.shape[-1] // resize_factor,
            ],
            InterpolationMode.BILINEAR,
            antialias=True,
        )

    def process_crops(self, uv_values, crops, original_crops, alpha=None):
        resized_crops = []
        cnn_output_crops = []
        render_dict = {"edit": [], "alpha": [], "edit_on_greenscreen": [], "composite": []}

        atlas_crop = crops[0]
        for i in range(3):
            grid_sampled_atlas_crop = F.grid_sample(
                atlas_crop,
                uv_values[i],
                mode="bilinear",
                align_corners=self.config["align_corners"],
            ).clamp(min=0.0, max=1.0)
            resized_crops.append(grid_sampled_atlas_crop)
        cnn_output = self.netG(atlas_crop)
        cnn_output_crops.append(cnn_output[:, :3])
        rendered_atlas_crops = self.render(cnn_output, bg_image=atlas_crop)
        for key, value in rendered_atlas_crops.items():
            for i in range(3):
                sampled_frame_crop = F.grid_sample(
                    value,
                    uv_values[i],
                    mode="bilinear",
                    align_corners=self.config["align_corners"],
                ).clamp(min=0.0, max=1.0)
                if alpha is not None:
                    sampled_frame_crop = sampled_frame_crop * alpha[i]
                    if key == "edit_on_greenscreen":
                        greenscreen = torch.zeros_like(sampled_frame_crop).to(sampled_frame_crop.device)
                        greenscreen[:, 1, :, :] = 177 / 255
                        greenscreen[:, 2, :, :] = 64 / 255
                        sampled_frame_crop += (1 - alpha[i]) * greenscreen
                render_dict[key].append(sampled_frame_crop.squeeze(0))

        # passing a random frame to the network
        frame_index = random.randint(0, 2)  # randomly sample one of three frames
        rec_crop = original_crops[frame_index]
        resized_crops.append(rec_crop)
        cnn_output = self.netG(rec_crop)
        if alpha is not None:
            alpha_crop = alpha[frame_index]
            cnn_output = cnn_output * alpha_crop
        cnn_output_crops.append(cnn_output[:, :3])

        rendered_frame_crop = self.render(cnn_output, bg_image=original_crops[frame_index])
        for key, value in rendered_frame_crop.items():
            render_dict[key].append(value.squeeze(0))

        return render_dict, resized_crops, cnn_output_crops

    def process_atlas(self, atlas):
        atlas_edit = self.netG(atlas)
        rendered_atlas = self.render(atlas_edit, bg_image=atlas)
        return rendered_atlas

    def forward(self, input_dict):
        inputs = input_dict["global_crops"]
        outputs = {"background": {}, "foreground": {}}

        if self.config["finetune_foreground"]:
            if self.config["multiply_foreground_alpha"]:
                alpha = inputs["foreground_alpha"]
            else:
                alpha = None
            foreground_outputs, resized_crops, cnn_output_crops = self.process_crops(
                inputs["foreground_uvs"],
                inputs["foreground_atlas_crops"],
                inputs["original_foreground_crops"],
                alpha=alpha,
            )
            outputs["foreground"]["output_crop"] = foreground_outputs
            outputs["foreground"]["cnn_inputs"] = resized_crops
            outputs["foreground"]["cnn_outputs"] = cnn_output_crops
            if "input_image" in input_dict.keys():
                outputs["foreground"]["output_image"] = self.process_atlas(input_dict["input_image"])
        elif self.config["finetune_background"]:
            background_outputs, resized_crops, cnn_output_crops = self.process_crops(
                inputs["background_uvs"],
                inputs["background_atlas_crops"],
                inputs["original_background_crops"],
            )
            outputs["background"]["output_crop"] = background_outputs
            outputs["background"]["cnn_inputs"] = resized_crops
            outputs["background"]["cnn_outputs"] = cnn_output_crops
            if "input_image" in input_dict.keys():
                outputs["background"]["output_image"] = self.process_atlas(input_dict["input_image"])
        return outputs

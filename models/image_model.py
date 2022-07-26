import torch
from .networks import define_G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.netG = define_G(cfg).to(device)

    def render(self, net_output, bg_image=None):
        assert net_output.min() >= 0 and net_output.max() <= 1
        edit = net_output[:, :3]
        alpha = net_output[:, 3].unsqueeze(1).repeat(1, 3, 1, 1)
        greenscreen = torch.zeros_like(edit).to(edit.device)
        greenscreen[:, 1, :, :] = 177 / 255
        greenscreen[:, 2, :, :] = 64 / 255
        edit_on_greenscreen = alpha * edit + (1 - alpha) * greenscreen
        outputs = {"edit": edit, "alpha": alpha, "edit_on_greenscreen": edit_on_greenscreen}
        if bg_image is not None:
            outputs["composite"] = (1 - alpha) * bg_image + alpha * edit

        return outputs

    def forward(self, input):
        outputs = {}
        # augmented examples
        if "input_crop" in input:
            outputs["output_crop"] = self.render(self.netG(input["input_crop"]), bg_image=input["input_crop"])

        # pass the entire image (w/o augmentations)
        if "input_image" in input:
            outputs["output_image"] = self.render(self.netG(input["input_image"]), bg_image=input["input_image"])

        # move outputs to list
        for outer_key in outputs.keys():
            for key, value in outputs[outer_key].items():
                outputs[outer_key][key] = [value[0]]

        return outputs

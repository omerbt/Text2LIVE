import torch.nn

from models.clip_extractor import ClipExtractor
from util.losses import LossG


class AtlasLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.clip_extractor = ClipExtractor(config)
        common_config = {
            key: config[key]
            for key in [
                "lambda_composition",
                "lambda_sparsity",
                "lambda_screen",
                "lambda_alpha_l1",
                "lambda_alpha_l0",
                "text_criterion",
                "clip_model_name",
                "bootstrap_epoch",
                "lambda_bootstrap",
                "relevancy_num_layers",
                "lambda_structure",
                "bootstrap_text",
                "bootstrap_scheduler",
                "bootstrapping_min_cover",
                "use_negative_bootstrap",
                "bootstrap_negative_text",
                "bootstrap_negative_map_threshold",
                "lambda_bootstrap_min",
                "device",
            ]
        }
        texts_config = {
            "screen_text": config["screen_text"],
            "comp_text": config["comp_text"],
            "src_text": config["src_text"],
        }
        common_config.update(texts_config)
        self.loss = LossG(common_config, self.clip_extractor)


        self.config = config

    def forward(self, outputs, inputs):
        losses = {}
        if self.config["finetune_background"]:
            inputs["input_crop"] = [el.squeeze(0) for el in outputs["background"]["cnn_inputs"]]
            losses["background"] = self.loss(outputs["background"], inputs)
        elif self.config["finetune_foreground"]:
            inputs["input_crop"] = [el.squeeze(0) for el in outputs["foreground"]["cnn_inputs"]]
            losses["foreground"] = self.loss(outputs["foreground"], inputs)
        return losses

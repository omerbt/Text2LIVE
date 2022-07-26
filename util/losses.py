import torch
import numpy as np
import torchvision.transforms as T

from models.clip_relevancy import ClipRelevancy
from util.aug_utils import RandomSizeCrop
from util.util import get_screen_template, get_text_criterion, get_augmentations_template


class LossG(torch.nn.Module):
    def __init__(self, cfg, clip_extractor):
        super().__init__()

        self.cfg = cfg

        # calculate target text embeddings
        template = get_augmentations_template()
        self.src_e = clip_extractor.get_text_embedding(cfg["src_text"], template)
        self.target_comp_e = clip_extractor.get_text_embedding(cfg["comp_text"], template)
        self.target_greenscreen_e = clip_extractor.get_text_embedding(cfg["screen_text"], get_screen_template())

        self.clip_extractor = clip_extractor
        self.text_criterion = get_text_criterion(cfg)

        if cfg["bootstrap_epoch"] > 0 and cfg["lambda_bootstrap"] > 0:
            self.relevancy_extractor = ClipRelevancy(cfg)
            self.relevancy_criterion = torch.nn.MSELoss()
            self.lambda_bootstrap = cfg["lambda_bootstrap"]

    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0

        all_outputs_composite = []
        all_outputs_greenscreen = []
        all_outputs_edit = []
        all_outputs_alpha = []
        all_inputs = []
        for out, ins in zip(["output_crop", "output_image"], ["input_crop", "input_image"]):
            if out not in outputs:
                continue
            all_outputs_composite += outputs[out]["composite"]
            all_outputs_greenscreen += outputs[out]["edit_on_greenscreen"]
            all_outputs_edit += outputs[out]["edit"]
            all_outputs_alpha += outputs[out]["alpha"]
            all_inputs += inputs[ins]

        # calculate alpha bootstrapping loss
        if inputs["step"] < self.cfg["bootstrap_epoch"] and self.cfg["lambda_bootstrap"] > 0:
            losses["loss_bootstrap"] = self.calculate_relevancy_loss(all_outputs_alpha, all_inputs)

            if self.cfg["bootstrap_scheduler"] == "linear":
                lambda_bootstrap = self.cfg["lambda_bootstrap"] * (
                    1 - (inputs["step"] + 1) / self.cfg["bootstrap_epoch"]
                )
            elif self.cfg["bootstrap_scheduler"] == "exponential":
                lambda_bootstrap = self.lambda_bootstrap * 0.99
                self.lambda_bootstrap = lambda_bootstrap
            elif self.cfg["bootstrap_scheduler"] == "none":
                lambda_bootstrap = self.lambda_bootstrap
            else:
                raise ValueError("Unknown bootstrap scheduler")
            lambda_bootstrap = max(lambda_bootstrap, self.cfg["lambda_bootstrap_min"])
            loss_G += losses["loss_bootstrap"] * lambda_bootstrap

        # calculate structure loss
        if self.cfg["lambda_structure"] > 0:
            losses["loss_structure"] = self.calculate_structure_loss(all_outputs_composite, all_inputs)
            loss_G += losses["loss_structure"] * self.cfg["lambda_structure"]

        # calculate composition loss
        if self.cfg["lambda_composition"] > 0:
            losses["loss_comp_clip"] = self.calculate_clip_loss(all_outputs_composite, self.target_comp_e)

            losses["loss_comp_dir"] = self.calculate_clip_dir_loss(
                all_inputs, all_outputs_composite, self.target_comp_e
            )

            loss_G += (losses["loss_comp_clip"] + losses["loss_comp_dir"]) * self.cfg["lambda_composition"]

        # calculate sparsity loss
        if self.cfg["lambda_sparsity"] > 0:
            total, l0, l1 = self.calculate_alpha_reg(all_outputs_alpha)
            losses["loss_sparsity"] = total
            losses["loss_sparsity_l0"] = l0
            losses["loss_sparsity_l1"] = l1

            loss_G += losses["loss_sparsity"] * self.cfg["lambda_sparsity"]

        # calculate screen loss
        if self.cfg["lambda_screen"] > 0:
            losses["loss_screen"] = self.calculate_clip_loss(all_outputs_greenscreen, self.target_greenscreen_e)
            loss_G += losses["loss_screen"] * self.cfg["lambda_screen"]

        losses["loss"] = loss_G
        return losses

    def calculate_alpha_reg(self, prediction):
        """
        Calculate the alpha sparsity term: linear combination between L1 and pseudo L0 penalties
        """
        l1_loss = 0.0
        for el in prediction:
            l1_loss += el.mean()
        l1_loss = l1_loss / len(prediction)
        loss = self.cfg["lambda_alpha_l1"] * l1_loss
        # Pseudo L0 loss using a squished sigmoid curve.
        l0_loss = 0.0
        for el in prediction:
            l0_loss += torch.mean((torch.sigmoid(el * 5.0) - 0.5) * 2.0)
        l0_loss = l0_loss / len(prediction)
        loss += self.cfg["lambda_alpha_l0"] * l0_loss
        return loss, l0_loss, l1_loss

    def calculate_clip_loss(self, outputs, target_embeddings):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]

        loss = 0.0
        for img in outputs:  # avoid memory limitations
            img_e = self.clip_extractor.get_image_embedding(img.unsqueeze(0))
            for target_embedding in target_embeddings:
                loss += self.text_criterion(img_e, target_embedding.unsqueeze(0))

        loss /= len(outputs) * len(target_embeddings)
        return loss

    def calculate_clip_dir_loss(self, inputs, outputs, target_embeddings):
        # randomly select embeddings
        n_embeddings = np.random.randint(1, min(len(self.src_e), len(target_embeddings)) + 1)
        idx = torch.randint(min(len(self.src_e), len(target_embeddings)), (n_embeddings,))
        src_embeddings = self.src_e[idx]
        target_embeddings = target_embeddings[idx]
        target_dirs = target_embeddings - src_embeddings

        loss = 0.0
        for in_img, out_img in zip(inputs, outputs):  # avoid memory limitations
            in_e = self.clip_extractor.get_image_embedding(in_img.unsqueeze(0))
            out_e = self.clip_extractor.get_image_embedding(out_img.unsqueeze(0))
            for target_dir in target_dirs:
                loss += 1 - torch.nn.CosineSimilarity()(out_e - in_e, target_dir.unsqueeze(0)).mean()

        loss /= len(outputs) * len(target_dirs)
        return loss

    def calculate_structure_loss(self, outputs, inputs):
        loss = 0.0
        for input, output in zip(inputs, outputs):
            with torch.no_grad():
                target_self_sim = self.clip_extractor.get_self_sim(input.unsqueeze(0))
            current_self_sim = self.clip_extractor.get_self_sim(output.unsqueeze(0))
            loss = loss + torch.nn.MSELoss()(current_self_sim, target_self_sim)
        loss = loss / len(outputs)
        return loss

    def calculate_relevancy_loss(self, alpha, input_img):
        positive_relevance_loss = 0.0
        for curr_alpha, curr_img in zip(alpha, input_img):
            x = torch.stack([curr_alpha, curr_img], dim=0)  # [2, 3, H, W]
            x = T.Compose(
                [
                    RandomSizeCrop(min_cover=self.cfg["bootstrapping_min_cover"]),
                    T.Resize((224, 224)),
                ]
            )(x)
            curr_alpha, curr_img = x[0].unsqueeze(0), x[1].unsqueeze(0)
            positive_relevance = self.relevancy_extractor(curr_img)
            positive_relevance_loss = self.relevancy_criterion(curr_alpha[0], positive_relevance.repeat(3, 1, 1))
            if self.cfg["use_negative_bootstrap"]:
                negative_relevance = self.relevancy_extractor(curr_img, negative=True)
                relevant_values = negative_relevance > self.cfg["bootstrap_negative_map_threshold"]
                negative_alpha_local = (1 - curr_alpha) * relevant_values.unsqueeze(1)
                negative_relevance_local = negative_relevance * relevant_values
                negative_relevance_loss = self.relevancy_criterion(
                    negative_alpha_local,
                    negative_relevance_local.unsqueeze(1).repeat(1, 3, 1, 1),
                )
                positive_relevance_loss += negative_relevance_loss
        positive_relevance_loss = positive_relevance_loss / len(alpha)
        return positive_relevance_loss

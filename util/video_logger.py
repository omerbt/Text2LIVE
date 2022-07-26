from pathlib import Path

import imageio
import torch

from util.util import tensor2im


class DataLogger:
    def __init__(self, config, dataset):
        self.layers_edits = {
            "background": dataset.original_video.detach().cpu(),
            "foreground": dataset.original_video.detach().cpu(),
        }
        self.alpha_video = dataset.all_alpha.detach().cpu()
        self.config = config
        self.layer_name = "foreground" if config["finetune_foreground"] else "background"

    @torch.no_grad()
    def log_data(self, epoch, lr, losses, model, dataset):
        log_data = {}
        for layer, layer_losses in losses.items():
            for key in layer_losses:
                log_data[f"Loss/{layer}_{key}"] = layer_losses[key].detach()
        log_data["epoch"] = epoch

        log_data["lr"] = lr

        if epoch % self.config["log_images_freq"] == 0:
            layer = self.layer_name
            edited_atlas_dict, edit_dict, uv_mask = dataset.render_video_from_atlas(model, layer=layer)
            alpha_of_edit = None
            edit_only = None
            for key in edited_atlas_dict.keys():
                if key != "edit":
                    masked = tensor2im(edited_atlas_dict[key].detach().cpu() * uv_mask)
                    log_data[f"Atlases/{layer}_masked_{key}"] = (
                        wandb.Image(masked) if self.config["use_wandb"] else masked
                    )
                if key == "alpha":
                    alpha_of_edit = edited_atlas_dict[key].detach().cpu() * uv_mask
                if key == "edit":
                    edit_only = edited_atlas_dict[key].detach().cpu() * uv_mask
            rgba_edit = tensor2im(torch.cat((edit_only, alpha_of_edit[:, [0]]), dim=1))
            log_data[f"Atlases/{layer}_rgba_layer"] = wandb.Image(rgba_edit) if self.config["use_wandb"] else rgba_edit

            for key in edit_dict.keys():
                if key != "composite" and key != "edit":
                    video = (255 * edit_dict[key].detach().cpu()).to(torch.uint8)
                    log_data[f"Videos/{layer}_{key}"] = (
                        wandb.Video(video, fps=10, format="mp4") if self.config["use_wandb"] else video
                    )

            if self.config[f"finetune_{layer}"]:
                self.layers_edits[layer] = edit_dict["composite"].detach().cpu()
            full_video = (
                self.alpha_video * self.layers_edits["foreground"]
                + (1 - self.alpha_video) * self.layers_edits["background"]
            )
            full_video = (255 * full_video.detach().cpu()).to(torch.uint8)
            log_data["Videos/full_video"] = (
                wandb.Video(full_video, fps=10, format="mp4") if self.config["use_wandb"] else full_video
            )

            # save model checkpoint
            if epoch > self.config["save_model_starting_epoch"]:
                filename = f"checkpoint_epoch_{epoch}.pt"
                dict_to_save = {
                    "model": model.state_dict(),
                }
                if self.config["use_wandb"]:
                    checkpoint_path = f"{wandb.run.dir}/{filename}"
                else:
                    checkpoint_path = f"{self.config['results_folder']}/{filename}"
                torch.save(dict_to_save, checkpoint_path)
        return log_data

    def save_locally(self, log_data):
        path = Path(self.config["results_folder"], str(log_data["epoch"]))
        path.mkdir(parents=True, exist_ok=True)
        for key in log_data.keys():
            save_name = key.replace("/", "_")
            if key.startswith("Videos"):
                imageio.mimwrite(f"{path}/{save_name}.mp4", log_data[key].permute(0, 2, 3, 1))
            elif key.startswith("Atlases"):
                imageio.imwrite(f"{path}/{save_name}.png", log_data[key])

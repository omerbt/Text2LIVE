import random
from argparse import ArgumentParser
import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.image_dataset import SingleImageDataset
from models.clip_extractor import ClipExtractor
from models.image_model import Model
from util.losses import LossG
from util.util import tensor2im, get_optimizer


def train_model(config):

    # set seed
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f"running with seed: {seed}.")

    # create dataset, loader
    dataset = SingleImageDataset(config)

    # define model
    model = Model(config)

    # define loss function
    clip_extractor = ClipExtractor(config)
    criterion = LossG(config, clip_extractor)

    # define optimizer, scheduler
    optimizer = get_optimizer(config, model.parameters())

    for epoch in tqdm(range(1, config["n_epochs"] + 1)):
        inputs = dataset[0]
        for key in inputs:
            if key != "step":
                inputs[key] = inputs[key].to(config["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        for key in inputs:
            if key != "step":
                inputs[key] = [inputs[key][0]]
        losses = criterion(outputs, inputs)
        loss_G = losses["loss"]
        log_data = losses
        log_data["epoch"] = epoch

        # log current generated image to wandb
        if epoch % config["log_images_freq"] == 0:
            src_img = dataset.get_img().to(config["device"])
            with torch.no_grad():
                output = model.render(model.netG(src_img), bg_image=src_img)
            for layer_name, layer_img in output.items():
                image_numpy_output = tensor2im(layer_img)
                log_data[layer_name] = [wandb.Image(image_numpy_output)] if config["use_wandb"] else image_numpy_output

        loss_G.backward()
        optimizer.step()

        # update learning rate
        if config["scheduler_policy"] == "exponential":
            optimizer.param_groups[0]["lr"] = max(config["min_lr"], config["gamma"] * optimizer.param_groups[0]["lr"])
        lr = optimizer.param_groups[0]["lr"]
        log_data["lr"] = lr

        if config["use_wandb"]:
            wandb.log(log_data)
        else:
            if epoch % config["log_images_freq"] == 0:
                save_locally(config["results_folder"], log_data)


def save_locally(results_folder, log_data):
    path = Path(results_folder, str(log_data["epoch"]))
    path.mkdir(parents=True, exist_ok=True)
    for key in log_data.keys():
        if key in ["composite", "alpha", "edit_on_greenscreen", "edit"]:
            imageio.imwrite(f"{path}/{key}.png", log_data[key])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/image_config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="golden_horse.yaml",
        help="Example config name",
    )
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(f"./configs/image_example_configs/{args.example_config}", "r") as f:
        example_config = yaml.safe_load(f)
    config["example_config"] = args.example_config
    config.update(example_config)

    run_name = f"-{config['image_path'].split('/')[-1]}"
    if config["use_wandb"]:
        import wandb

        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, name=run_name)
        wandb.run.name = str(wandb.run.id) + wandb.run.name
        config = dict(wandb.config)
    else:
        now = datetime.datetime.now()
        run_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}{run_name}"
        path = Path(f"{config['results_folder']}/{run_name}")
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)
        config["results_folder"] = str(path)

    train_model(config)
    if config["use_wandb"]:
        wandb.finish()
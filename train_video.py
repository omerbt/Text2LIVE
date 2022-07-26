import datetime
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.video_dataset import AtlasDataset
from models.video_model import VideoModel
from util.atlas_loss import AtlasLoss
from util.util import get_optimizer
from util.video_logger import DataLogger


def train_model(config):
    # set seed
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"running with seed: {seed}.")

    dataset = AtlasDataset(config)
    model = VideoModel(config)
    criterion = AtlasLoss(config)
    optimizer = get_optimizer(config, model.parameters())

    logger = DataLogger(config, dataset)
    with tqdm(range(1, config["n_epochs"] + 1)) as tepoch:
        for epoch in tepoch:
            inputs = dataset[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, inputs)

            loss = 0.
            if config["finetune_foreground"]:
                loss += losses["foreground"]["loss"]
            elif config["finetune_background"]:
                loss += losses["background"]["loss"]

            lr = optimizer.param_groups[0]["lr"]
            log_data = logger.log_data(epoch, lr, losses, model, dataset)

            loss.backward()
            optimizer.step()
            optimizer.param_groups[0]["lr"] = max(config["min_lr"], config["gamma"] * optimizer.param_groups[0]["lr"])

            if config["use_wandb"]:
                wandb.log(log_data)
            else:
                if epoch % config["log_images_freq"] == 0:
                    logger.save_locally(log_data)

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=loss.item())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/video_config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="car-turn_winter.yaml",
        help="Example config name",
    )
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(f"./configs/video_example_configs/{args.example_config}", "r") as f:
        example_config = yaml.safe_load(f)
    config["example_config"] = args.example_config
    config.update(example_config)

    run_name = f"-{config['checkpoint_path'].split('/')[-2]}"
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

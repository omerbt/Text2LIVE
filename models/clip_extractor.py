import torch
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from CLIP import clip

from util.util import compose_text_with_templates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = clip.load(cfg["clip_model_name"], device=device)[0]
        self.model = model.eval().requires_grad_(False)

        self.clip_input_size = 224
        self.clip_normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.basic_transform = T.Compose(
            [
                # we added interpolation to CLIP positional embedding, allowing to work with arbitrary resolution.
                T.Resize(self.clip_input_size, max_size=380),
                self.clip_normalize,
            ]
        )
        # list of augmentations we apply before calculating the CLIP losses
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=15,
                            translate=(0.1, 0.1),
                            fill=cfg["clip_affine_transform_fill"],
                            interpolation=InterpolationMode.BILINEAR,
                        )
                    ],
                    p=0.8,
                ),
                T.RandomPerspective(
                    distortion_scale=0.4,
                    p=0.5,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=cfg["clip_affine_transform_fill"],
                ),
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.7),
                T.RandomGrayscale(p=0.15),
            ]
        )

        self.n_aug = cfg["n_aug"]

    def augment_input(self, input, n_aug=None, clip_input_size=None):
        if n_aug is None:
            n_aug = self.n_aug
        if clip_input_size is None:
            clip_input_size = self.clip_input_size

        cutouts = []
        cutout = T.Resize(clip_input_size, max_size=320)(input)
        cutout_h, cutout_w = cutout.shape[-2:]
        cutout = self.augs(cutout)
        cutouts.append(cutout)
        sideY, sideX = input.shape[2:4]
        for _ in range(n_aug - 1):
            s = (
                torch.zeros(
                    1,
                )
                .uniform_(0.6, 1)
                .item()
            )
            h = int(sideY * s)
            w = int(sideX * s)
            cutout = T.RandomCrop(size=(h, w))(input)
            cutout = T.Resize((cutout_h, cutout_w))(cutout)
            cutout = self.augs(cutout)
            cutouts.append(cutout)

        cutouts = torch.cat(cutouts)
        return cutouts

    def get_image_embedding(self, x, aug=True):
        if aug:
            views = self.augment_input(x)
        else:
            views = self.basic_transform(x)
        if type(views) == list:
            image_embeds = []
            for view in views:
                image_embeds.append(self.encode_image(self.clip_normalize(view)))
            image_embeds = torch.cat(image_embeds)
        else:
            image_embeds = self.encode_image(self.clip_normalize(views))
        return image_embeds

    def encode_image(self, x):
        return self.model.encode_image(x)

    def get_text_embedding(self, text, template, average_embeddings=False):
        if type(text) == str:
            text = [text]
        embeddings = []
        for prompt in text:
            with torch.no_grad():
                embedding = self.model.encode_text(
                    clip.tokenize(compose_text_with_templates(prompt, template)).to(device)
                )
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)
        if average_embeddings:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        return embeddings

    def get_self_sim(self, x):
        x = self.basic_transform(x)
        return self.model.calculate_self_sim(x)

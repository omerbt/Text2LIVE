import torch
from torchvision import transforms as T
import numpy as np
from CLIP import clip_explainability as clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb
class ClipRelevancy(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # TODO it would make more sense not to load ths model again (already done in the extractor)
        self.model = clip.load("ViT-B/32", device=device, jit=False)[0]
        clip_input_size = 224
        self.preprocess = T.Compose(
            [
                T.Resize((clip_input_size, clip_input_size)),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        input_prompts = cfg["bootstrap_text"]
        if type(input_prompts) == str:
            input_prompts = [input_prompts]
        self.text = clip.tokenize(input_prompts).to(cfg["device"])

        if self.cfg["use_negative_bootstrap"]:
            input_negative_prompts = cfg["bootstrap_negative_text"]
            if type(input_negative_prompts) == str:
                input_negative_prompts = [input_negative_prompts]
            self.bootstrap_negative_text = clip.tokenize(input_negative_prompts).to(cfg["device"])

    def image_relevance(self, image_relevance):
        patch_size = 32  # hardcoded for ViT-B/32 which we use
        h = w = 224
        image_relevance = image_relevance.reshape(1, 1, h // patch_size, w // patch_size)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=(h, w), mode="bilinear")
        image_relevance = image_relevance.reshape(h, w).to(device)
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        return image_relevance

    def interpret(self, image, negative=False):
        text = self.text if not negative else self.bootstrap_negative_text
        batch_size = text.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        # TODO this is pretty inefficient, we can calculate the text embeddings instead of recomputing at each call
        logits_per_image, logits_per_text = self.model(images, text)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <= self.cfg["relevancy_num_layers"]:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        return image_relevance

    def forward(self, img, preprocess=True, negative=False):
        if preprocess:
            img = self.preprocess(img)
        R_image = self.interpret(img, negative=negative)
        res = []
        for el in R_image:
            res.append(self.image_relevance(el).float())
        res = torch.stack(res, dim=0)
        return res

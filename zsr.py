"""
Testing the zero-shot recognition capability
of CLIP models on HOI datasets

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import os
import clip
import json
import torch
import pocket
import argparse

from torchvision.transforms import (
    Compose, Resize, CenterCrop,
    ToTensor, Normalize,
    InterpolationMode
)

from tqdm import tqdm
from hicodet.hicodet import HICODet

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def clip_forward(ctx, x):
    x = x.type(ctx.visual.conv1.weight.dtype)
    x = ctx.visual.relu1(ctx.visual.bn1(ctx.visual.conv1(x)))
    x = ctx.visual.relu2(ctx.visual.bn2(ctx.visual.conv2(x)))
    x = ctx.visual.relu3(ctx.visual.bn3(ctx.visual.conv3(x)))
    x = ctx.visual.avgpool(x)

    x = ctx.visual.layer1(x)
    x = ctx.visual.layer2(x)
    x = ctx.visual.layer3(x)
    x = ctx.visual.layer4(x)
    x = Resize(7, InterpolationMode.BICUBIC)(x)
    x = ctx.visual.attnpool(x)

    return x

@torch.no_grad()
def main(args):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    def target_transform(t):
        labels = torch.zeros(600)
        labels[t["hoi"]] = 1
        return labels
    dataset = HICODet(
        os.path.join(args.data_root, f"hico_20160224_det/images/{args.partition}"),
        os.path.join(args.data_root, f"instances_{args.partition}.json"),
        transform=_transform(224), target_transform=target_transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=use_cuda,
    )

    with open("hicodet/prompts.json", "r") as f:
        prompts = json.load(f)
    text = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=1, keepdim=True)

    if args.map:
        eval_zs_recognition(dataloader, model, text_features, device)

def eval_zs_recognition(dataloader, model, class_embeddings, device):
    """Evaluate the zero-shot interaction recognition mAP"""
    ap = pocket.utils.AveragePrecisionMeter()
    for images, targets in tqdm(dataloader):
        # image_features = model.encode_image(images.to(device))
        image_features = clip_forward(model, images.to(device))
        image_features /= image_features.norm(dim=1, keepdim=True)

        cos = image_features @ class_embeddings.t()
        ap.append(cos, targets)

    ap = ap.eval()
    print(
        f"The mAP is {ap.mean():.4f},"
        f" rare: {ap[dataloader.dataset.rare].mean():.4f},"
        f" none-rare: {ap[dataloader.dataset.non_rare].mean():.4f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str, default="hicodet")
    parser.add_argument("--partition", "-p", type=str, default="test2015")
    parser.add_argument("--batch-size", "-bs", type=int, default=128)
    parser.add_argument("--num-workers", "-nw", type=int, default=8)
    parser.add_argument("--map", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
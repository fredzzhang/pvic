"""
Testing the zero-shot recognition capability
of CLIP models on HOI datasets

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import os
import clip
import torch
import pocket
import argparse

from tqdm import tqdm
from hicodet.hicodet import HICODet

def construct_prompt(s):
    v, o = s.split()
    if v == "no_interaction":
        return f"a photo of a person and a {o}"
    else:
        return f"a photo of a person {v} a {o}"

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
        transform=preprocess, target_transform=target_transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=use_cuda,
        
    )

    text = clip.tokenize([construct_prompt(p) for p in dataset.interactions]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=1, keepdim=True)

    ap = pocket.utils.AveragePrecisionMeter()
    for images, targets in tqdm(dataloader):
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=1, keepdim=True)

        cos = image_features @ text_features.t()
        ap.append(cos, targets)

    print(ap.eval().mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str, default="hicodet")
    parser.add_argument("--partition", "-p", type=str, default="test2015")
    parser.add_argument("--batch-size", "-bs", type=int, default=128)
    parser.add_argument("--num-workers", "-nw", type=int, default=8)
    args = parser.parse_args()
    main(args)
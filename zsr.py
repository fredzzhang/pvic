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
    # x = Resize(7, InterpolationMode.BICUBIC)(x)
    x = ctx.visual.attnpool(x)

    return x

@torch.no_grad()
def main(args):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    # Bilinear interpolation on the positional encodings
    if args.resolution != 224:
        size = args.resolution // 32
        pe_cls, pe = model.visual.attnpool.positional_embedding.split([1, 49], dim=0)
        pe = pe.reshape(1, 7, 7, 2048).permute(0, 3, 1, 2)
        pe = Resize(size, InterpolationMode.BILINEAR)(pe).permute(0, 2, 3, 1).reshape(size ** 2, 2048)
        model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat([pe_cls, pe], dim=0))

    def target_transform(t):
        labels = torch.zeros(600)
        labels[t["hoi"]] = 1
        return labels
    dataset = HICODet(
        os.path.join(args.data_root, f"hico_20160224_det/images/{args.partition}"),
        os.path.join(args.data_root, f"instances_{args.partition}.json"),
        transform=_transform(args.resolution), target_transform=target_transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=use_cuda,
        shuffle=False
    )

    with open("hicodet/prompts.json", "r") as f:
        prompts = json.load(f)
    text = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=1, keepdim=True)

    if args.map:
        eval_zs_recognition(dataloader, model, text_features, device)
    if args.rec:
        eval_zs_recall(dataloader, model, text_features, device)
    if args.top_k:
        find_top_k_triplets(dataloader, model, text_features, device)

@torch.no_grad()
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

@torch.no_grad()
def eval_zs_recall(dataloader, model, class_embeddings, device):
    """Evaluate the recall from top k interactions with varying k"""
    full = torch.zeros(1, 600)
    rec = torch.zeros(8, 600)
    nuniqobj = torch.zeros(8, 2)
    nuniqobj_all = []
    intr2obj = torch.as_tensor(dataloader.dataset.class_corr)[:, 1]
    for images, targets in tqdm(dataloader):
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=1, keepdim=True)

        cos = image_features @ class_embeddings.t()
        rank = cos.argsort(dim=1, descending=True).cpu()
        i, j = torch.nonzero(targets).unbind(1)
        # Record the number of unique object types from the top-k triplets
        selection = rank[:, :150]
        obj_list = intr2obj[selection].unbind(0)
        for objs in obj_list:
            nuniqobj[0, 0] += objs[:3].unique().numel()
            nuniqobj[1, 0] += objs[:5].unique().numel()
            nuniqobj[2, 0] += objs[:10].unique().numel()
            nuniqobj[3, 0] += objs[:15].unique().numel()
            nuniqobj[4, 0] += objs[:25].unique().numel()
            nuniqobj[5, 0] += objs[:50].unique().numel()
            nuniqobj[6, 0] += objs[:100].unique().numel()
            nuniqobj[7, 0] += objs.unique().numel()
            nuniqobj_all.append(objs.unique().numel())
        nuniqobj[:, 1] += len(obj_list)
        # Duplicate the rank once for each ground truth instance
        ref = rank[i]
        isin = torch.eq(ref[:, :150], j.unsqueeze(1))
        isin3 = isin[:, :3].sum(1)
        isin5 = isin[:, :5].sum(1)
        isin10 = isin[:, :10].sum(1)
        isin15 = isin[:, :20].sum(1)
        isin25 = isin[:, :25].sum(1)
        isin50 = isin[:, :50].sum(1)
        isin100 = isin[:, :100].sum(1)
        isin150 = isin.sum(1)
        isin_ = torch.stack([isin3, isin5, isin10, isin15, isin25, isin50, isin100, isin150], dim=0)
        # Aggregate results for each ground truth instance
        for n, t in enumerate(j):
            rec[:, t] += isin_[:, n]
            full[0, t] += 1

    nuniqobj = nuniqobj[:, 0] / nuniqobj[:, 1]
    rec /= full
    print(
        f"Recall for top-k interactions:\n"
        f"k=3,\tfull: {rec[0].mean():.4f}, rare: {rec[0, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[0, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[0]:.2f}.\n"
        f"k=5,\tfull: {rec[1].mean():.4f}, rare: {rec[1, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[1, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[1]:.2f}.\n"
        f"k=10,\tfull: {rec[2].mean():.4f}, rare: {rec[2, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[2, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[2]:.2f}.\n"
        f"k=15,\tfull: {rec[3].mean():.4f}, rare: {rec[3, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[3, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[3]:.2f}.\n"
        f"k=25,\tfull: {rec[4].mean():.4f}, rare: {rec[4, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[4, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[4]:.2f}.\n"
        f"k=50,\tfull: {rec[5].mean():.4f}, rare: {rec[5, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[5, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[5]:.2f}.\n"
        f"k=100,\tfull: {rec[6].mean():.4f}, rare: {rec[6, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[6, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[6]:.2f}.\n"
        f"k=150,\tfull: {rec[7].mean():.4f}, rare: {rec[7, dataloader.dataset.rare].mean():.4f}, "
        f"non-rare: {rec[7, dataloader.dataset.non_rare].mean():.4f}, avg.#uniq. objs.: {nuniqobj[7]:.2f}.\n"
    )

    with open(f"nuniqobj_{len(nuniqobj_all)}.json", "w") as f:
        json.dump(nuniqobj_all, f)

@torch.no_grad()
def find_top_k_triplets(dataloader, model, class_embeddings, device, k=100):
    """Extract the top-k triplets"""
    top_k_t = []
    for images, _ in tqdm(dataloader):
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=1, keepdim=True)

        cos = image_features @ class_embeddings.t()
        rank = cos.argsort(dim=1, descending=True).cpu()
        top_k_t.append(rank[:, :k])
    top_k_t = torch.cat(top_k_t)
    torch.save(top_k_t, f"top_k_triplets_{len(top_k_t)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str, default="hicodet")
    parser.add_argument("--partition", "-p", type=str, default="test2015")
    parser.add_argument("--batch-size", "-bs", type=int, default=128)
    parser.add_argument("--resolution", "-r", type=int, default=224)
    parser.add_argument("--num-workers", "-nw", type=int, default=8)
    parser.add_argument("--map", action="store_true", default=False)
    parser.add_argument("--rec", action="store_true", default=False)
    parser.add_argument("--top-k", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
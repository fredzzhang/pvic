from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor
from ops import compute_sinusoidal_pe
import math
import clip
import numpy as np
import seaborn as sns

import cv2
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pocket

from hicodet.hicodet import HICODet
import torchvision.transforms.functional as TF

from PIL import Image

def plot_pe_attn():
    mask = torch.zeros(1, 50, 50).bool()
    src = torch.rand(1, 256, 50, 50)

    pe = PositionEmbeddingSine(128, temperature=20, normalize=True)
    kv_pe_w = torch.rand(256, 256)
    kv_pe = pe(NestedTensor(src, mask))

    points = torch.as_tensor([[[0.5, 0.5]]])
    q_pe_w = torch.rand(256, 256)
    q_pe = compute_sinusoidal_pe(points, temperature=20)
    q_pe[..., :128] *= .5
    q_pe[..., 128:] *= 2

    print(f"x={points[0, 0, 0]}, y={points[0, 0, 1]}")

    kv_pe = kv_pe.flatten(2).permute(0, 2, 1)

    # Apply linear transformation
    # kv_pe = kv_pe.matmul(kv_pe_w)
    # q_pe = q_pe.matmul(q_pe_w)

    dp = torch.matmul(kv_pe, q_pe.transpose(1, 2)) / 16
    dp = dp.squeeze(2).softmax(-1).reshape(-1, 50, 50)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(dp[0], cmap="Blues")
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig("attn.png")

def vis_attn_weights():

    rgb1 = sns.color_palette(n_colors=2)[1]
    # rgb2 = sns.color_palette("bright", n_colors=2)[0]
    rgb2 = (254 / 255, 254 / 255, 114 / 255)

    cdict = {
        "red": ((0., 0., 0.),
                (.5, rgb1[0], rgb1[0]),
                (1., rgb2[0], rgb2[0])),
        "green": ((0., 0., 0.),
                  (.5, rgb1[1], rgb1[1]),
                  (1., rgb2[1], rgb2[1])),
        "blue": ((0., 0., 0.),
                 (.5, rgb1[2], rgb1[2]),
                 (1., rgb2[2], rgb2[2])),
        "alpha":
                ((0., 0., 0.,),
                 (1., .8, .8,))
    }
    my_cmap = LinearSegmentedColormap("MyCMap", cdict)

    attn = torch.load("r50_dec2_attn_7680.pt")
    dets = torch.load("r50_dec2_dets_7680.pt")

    dataset = HICODet(
        "hicodet/hico_20160224_det/images/test2015",
        "hicodet/instances_test2015.json"
    )
    image, _ = dataset[7680]

    idx = 61

    scale = image.size[1] / dets[0]["size"][0]
    bh, bo = dets[0]["boxes"][dets[0]["pairing"][idx]] * scale

    pocket.utils.draw_boxes(image, torch.stack([bh, bo]), width=4)
    # image.save("box_pair.png")

    h, w = (dets[0]["size"] / 32).ceil().long()
    attn_map = attn[0, 7, dets[0]["x"][idx]].reshape(1, 1, h, w)
    attn_map = torch.nn.functional.interpolate(attn_map, size=(image.height, image.width),
                                               mode="bilinear", align_corners=True
                                               )
    plt.imshow(image)
    attn_image = attn_map[0, 0]
    ax = plt.gca()
    sns.heatmap(attn_image.numpy(), alpha=0.6, ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"attn_64.png", bbox_inches="tight", pad_inches=0)
    plt.close()

def vis_detr_attn_weights():
    rgb1 = sns.color_palette(n_colors=2)[1]
    # rgb2 = sns.color_palette("bright", n_colors=2)[0]
    rgb2 = (254 / 255, 254 / 255, 114 / 255)

    cdict = {
        "red": ((0., 0., 0.),
                (.5, rgb1[0], rgb1[0]),
                (1., rgb2[0], rgb2[0])),
        "green": ((0., 0., 0.),
                  (.5, rgb1[1], rgb1[1]),
                  (1., rgb2[1], rgb2[1])),
        "blue": ((0., 0., 0.),
                 (.5, rgb1[2], rgb1[2]),
                 (1., rgb2[2], rgb2[2])),
        "alpha":
                ((0., 0., 0.,),
                 (1., .8, .8,))
    }
    my_cmap = LinearSegmentedColormap("MyCMap", cdict)

    attn = torch.load("detr_attn.pt")

    keep = [61, 87, 71, 92, 12, 51]
    attn = attn[0, keep]

    dets = torch.load("dets.pth", map_location="cpu")
    dataset = HICODet(
        "hicodet/hico_20160224_det/images/test2015",
        "hicodet/instances_test2015.json"
    )
    image, _ = dataset[4050]

    idx = 33

    scale = image.size[1] / dets[0]["size"][0]

    bh, bo = dets[0]["boxes"][dets[0]["pairing"][idx]] * scale

    h, w = (dets[0]["size"] / 32).ceil().long()

    image_h = image.copy(); image_o = image.copy()
    pocket.utils.draw_boxes(image_h, bh[None], width=3)
    pocket.utils.draw_boxes(image_o, bo[None], width=3)

    bh_attn, bo_attn = attn[dets[0]["pairing"][idx]]
    bh_attn = bh_attn.reshape(1, 1, h, w)
    bo_attn = bo_attn.reshape(1, 1, h, w)

    bh_attn = torch.nn.functional.interpolate(bh_attn, size=(image.height, image.width),
                                              mode="bilinear", align_corners=True)
    bo_attn = torch.nn.functional.interpolate(bo_attn, size=(image.height, image.width),
                                              mode="bilinear", align_corners=True)

    plt.imshow(image_h)
    attn_image = bh_attn[0, 0]
    ax = plt.gca()
    sns.heatmap(attn_image.numpy(), alpha=0.6, ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"human_attn.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    plt.imshow(image_o)
    attn_image = bo_attn[0, 0]
    ax = plt.gca()
    sns.heatmap(attn_image.numpy(), alpha=0.6, ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"object_attn.png", bbox_inches="tight", pad_inches=0)
    plt.close()

def visualise_qpic_attn_weights():
    rgb1 = sns.color_palette(n_colors=2)[1]
    # rgb2 = sns.color_palette("bright", n_colors=2)[0]
    rgb2 = (254 / 255, 254 / 255, 114 / 255)

    cdict = {
        "red": ((0., 0., 0.),
                (.5, rgb1[0], rgb1[0]),
                (1., rgb2[0], rgb2[0])),
        "green": ((0., 0., 0.),
                  (.5, rgb1[1], rgb1[1]),
                  (1., rgb2[1], rgb2[1])),
        "blue": ((0., 0., 0.),
                 (.5, rgb1[2], rgb1[2]),
                 (1., rgb2[2], rgb2[2])),
        "alpha":
                ((0., 0., 0.,),
                 (1., .8, .8,))
    }
    my_cmap = LinearSegmentedColormap("MyCMap", cdict)

    attn = torch.load("qpic_attn.pt", map_location="cpu").detach()

    dets = torch.load("qpic_dets.pt", map_location="cpu")
    dataset = HICODet(
        "hicodet/hico_20160224_det/images/test2015",
        "hicodet/instances_test2015.json"
    )
    image, _ = dataset[4050]

    idx = 38
    bh = dets["boxes"][dets["sub_ids"][idx]]
    bo = dets["boxes"][dets["obj_ids"][idx]]

    pocket.utils.draw_boxes(image, torch.stack([bh, bo]), width=3)

    vb_score = dets["verb_scores"][idx][111]
    attn = attn[0, idx].reshape(1, 1, 25, 38)
    attn = torch.nn.functional.interpolate(attn, size=(image.height, image.width),
                                              mode="bilinear", align_corners=True)
    plt.imshow(image)
    attn_image = attn[0, 0]
    ax = plt.gca()
    sns.heatmap(attn_image.numpy(), alpha=0.6, ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"qpic_attn_{vb_score:.2f}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def vis_all_attn_weights():

    rgb1 = sns.color_palette(n_colors=2)[1]
    # rgb2 = sns.color_palette("bright", n_colors=2)[0]
    rgb2 = (254 / 255, 254 / 255, 114 / 255)

    cdict = {
        "red": ((0., 0., 0.),
                (.5, rgb1[0], rgb1[0]),
                (1., rgb2[0], rgb2[0])),
        "green": ((0., 0., 0.),
                  (.5, rgb1[1], rgb1[1]),
                  (1., rgb2[1], rgb2[1])),
        "blue": ((0., 0., 0.),
                 (.5, rgb1[2], rgb1[2]),
                 (1., rgb2[2], rgb2[2])),
        "alpha":
                ((0., 0., 0.,),
                 (1., .8, .8,))
    }
    my_cmap = LinearSegmentedColormap("MyCMap", cdict)

    attn = torch.load("r50_dec2_attn_4050.pt")
    c_and_p = attn["c_and_p"]
    c = attn["c"]; p = attn["p"]
    cp = attn["cp"]; pc = attn["pc"]

    dets = torch.load("dets.pth", map_location="cpu")
    dataset = HICODet(
        "hicodet/hico_20160224_det/images/test2015",
        "hicodet/instances_test2015.json"
    )
    image, _ = dataset[4050]

    idx = 33

    scale = image.size[1] / dets[0]["size"][0]
    bh, bo = dets[0]["boxes"][dets[0]["pairing"][idx]] * scale

    pocket.utils.draw_boxes(image, torch.cat([bh[None], bo[None]]), width=3)
    # image.save("box_pair.png")

    h, w = (dets[0]["size"] / 32).ceil().long()

    c_and_p = c_and_p[:, :, dets[0]["x"][idx]].reshape(1, 8, h, w)
    c_and_p = torch.nn.functional.interpolate(c_and_p, size=(image.height, image.width),
                                              mode="bilinear", align_corners=True)
    for i in range(8):
        plt.imshow(image)
        attn_image = c_and_p[0, i]
        ax = plt.gca()
        sns.heatmap(attn_image.numpy(), alpha=0.6, ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"attn/c_and_p_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    c = c[:, :, dets[0]["x"][idx]].reshape(1, 8, h, w)
    c = torch.nn.functional.interpolate(c, size=(image.height, image.width),
                                        mode="bilinear", align_corners=True)
    for i in range(8):
        plt.imshow(image)
        attn_image = c[0, i]
        ax = plt.gca()
        sns.heatmap(attn_image.numpy(), alpha=0.6,  ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"attn/c_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    p = p[:, :, dets[0]["x"][idx]].reshape(1, 8, h, w)
    p = torch.nn.functional.interpolate(p, size=(image.height, image.width),
                                        mode="bilinear", align_corners=True)
    for i in range(8):
        plt.imshow(image)
        attn_image = p[0, i]
        # x, y = torch.nonzero(attn_image < 5e-4).unbind(1)
        # attn_image[x, y] = 0
        ax = plt.gca()
        sns.heatmap(attn_image.numpy(), alpha=0.6,  ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"attn/p_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    cp = cp[:, :, dets[0]["x"][idx]].reshape(1, 8, h, w)
    cp = torch.nn.functional.interpolate(cp, size=(image.height, image.width),
                                        mode="bilinear", align_corners=True)
    for i in range(8):
        plt.imshow(image)
        attn_image = cp[0, i]
        ax = plt.gca()
        sns.heatmap(attn_image.numpy(), alpha=0.6,  ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"attn/cp_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    pc = pc[:, :, dets[0]["x"][idx]].reshape(1, 8, h, w)
    pc = torch.nn.functional.interpolate(pc, size=(image.height, image.width),
                                        mode="bilinear", align_corners=True)
    for i in range(8):
        plt.imshow(image)
        attn_image = pc[0, i]
        ax = plt.gca()
        sns.heatmap(attn_image.numpy(), alpha=0.6,  ax=ax, xticklabels=False, yticklabels=False, cbar=False, cmap=my_cmap)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"attn/pc_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    # image.save("detection.png")

def vis_all_attn_weights_manually():
    attn = torch.load("qk.pth", map_location='cpu')
    dets = torch.load("dets.pth", map_location='cpu')
    avg_attn = torch.load("attn.pth", map_location="cpu")

    dataset = HICODet(
        "hicodet/hico_20160224_det/images/test2015",
        "hicodet/instances_test2015.json"
    )
    image, _ = dataset[998]

    idx = 111

    scale = image.size[1] / dets[0]["size"][0]
    bh, bo = dets[0]["boxes"][dets[0]["pairing"][idx]] * scale

    pocket.utils.draw_box_pairs(image, bh[None], bo[None])
    # image.save("box_pair.png")

    p_idx = dets[0]["x"][idx]
    h, w = (dets[0]["size"] / 32).ceil().long()

    q = attn["q"]; k = attn["k"]
    q_p = attn["q_p"]; k_p = attn["k_p"]
    attn_c = torch.bmm(
        q.view(-1, 8, 48).permute(1, 0, 2),
        k.view(-1, 8, 48).permute(1, 2, 0)
    ) / (96 ** -0.5)
    attn_p = torch.bmm(
        q_p.view(-1, 8, 48).permute(1, 0, 2),
        k_p.view(-1, 8, 48).permute(1, 2, 0)
    ) / (96 ** -0.5)
    attn_combine = attn_c + attn_p

    attn_c = attn_c.softmax(-1)[:, p_idx].view(8, 1, h, w)
    attn_p = attn_p.softmax(-1)[:, p_idx].view(8, 1, h, w)
    attn_combine = attn_combine.softmax(-1)[:, p_idx].view(8, 1, h, w)
    avg_attn = avg_attn[0, p_idx].view(1, 1, h, w)

    attn_c = torch.nn.functional.interpolate(attn_c, size=(image.height, image.width), mode="bilinear")
    attn_p = torch.nn.functional.interpolate(attn_p, size=(image.height, image.width), mode="bilinear")
    attn_combine = torch.nn.functional.interpolate(attn_combine, size=(image.height, image.width), mode="bilinear")
    avg_attn = torch.nn.functional.interpolate(avg_attn, size=(image.height, image.width), mode="bilinear") * 10

    def organise_in_grid(x, h, w):
        assert h * w == x.shape[0]
        ih, iw = x.shape[1:]
        x = x.split(w, 0)
        x = torch.cat([x_.permute(1, 0, 2).reshape(ih, iw * w) for x_ in x], dim=0)
        return x
    def convert_single_channel_mask_to_image(x, c=0):
        assert x.shape[0] == 1 and x.ndim == 3
        image = torch.zeros(3, *x.shape[1:])
        image[c] = x
        return image

    attn_c = organise_in_grid(attn_c.squeeze(1), 2, 4).unsqueeze(0)
    attn_p = organise_in_grid(attn_p.squeeze(1), 2, 4).unsqueeze(0)
    attn_combine = organise_in_grid(attn_combine.squeeze(1), 2, 4).unsqueeze(0)

    attn_c = convert_single_channel_mask_to_image(attn_c, c=0)
    attn_p = convert_single_channel_mask_to_image(attn_p, c=0)
    attn_combine = convert_single_channel_mask_to_image(attn_combine, c=0)
    avg_attn = convert_single_channel_mask_to_image(avg_attn.squeeze(0), c=0)

    image_avg_attn = Image.blend(TF.to_pil_image(avg_attn), image, alpha=0.2)
    image_avg_attn.save("avg_attn.png")

    image = TF.to_tensor(image)
    image = image.repeat(1, 2, 4)

    image_c = TF.to_pil_image(image)
    image_p = TF.to_pil_image(image)
    image_combine = TF.to_pil_image(image)

    image_c = Image.blend(TF.to_pil_image(attn_c), image_c, alpha=0.2)
    image_p = Image.blend(TF.to_pil_image(attn_p), image_p, alpha=0.2)
    image_combine = Image.blend(TF.to_pil_image(attn_combine), image_combine, alpha=0.2)

    image_c.save("content_attn.png")
    image_p.save("positional_attn.png")
    image_combine.save("combined_attn.png")

def generate_box_binary_mask():
    boxes = [torch.as_tensor([
        [20., 120., 430., 390.],
        [45.7, 201.6, 298.3, 512.4],
    ])]
    image_sizes = torch.as_tensor([[480, 640]]).long()
    dest_sizes = torch.as_tensor([[15, 20]]).long()

    j = 0
    for bx, in_sz, out_sz in zip(boxes, image_sizes, dest_sizes):
        bx = bx * out_sz[[1, 0]].repeat(2) / in_sz[[1, 0]].repeat(2)
        x1, y1, x2, y2 = bx.unbind(1)
        x1 = x1.floor().long()
        y1 = y1.floor().long()
        x2 = x2.ceil().long()
        y2 = y2.ceil().long()

        m = torch.ones(len(x1), *out_sz.tolist(), device=bx.device, dtype=torch.bool)
        for i in range(len(m)):
            m[i, y1[i]: y2[i], x1[i]: x2[i]] = False
        j += 1

        merge = m.prod(0).bool()
        image = TF.to_pil_image(merge.unsqueeze(0).repeat(3, 1, 1).float())
        # pocket.utils.draw_boxes(image, bx, outline=(255, 0, 0))
        image.save("mask.png")

def show_masks():
    h, w = 21, 28
    a = torch.load("masks.pth")
    for i, m in enumerate(a):
        m = m.view(h, w)
        plt.imshow(m.numpy(), cmap="Blues")
        plt.savefig(f"mask_{i}_{m.sum().item()}.png")
        plt.close()

def test_clip(images, vlm, targets):
    x = images.type(vlm.dtype)
    x = vlm.visual.conv1(x)                # bs, c, h, w
    x = x.reshape(x.shape[0], x.shape[1], -1)   # bs, c, hw
    x = x.permute(0, 2, 1)                      # bs, hw, c
    x = torch.cat([vlm.visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    ), x], dim=1)                               # bs, hw + 1, c
    x = x + vlm.visual.positional_embedding.to(x.dtype)
    x = vlm.visual.ln_pre(x)

    x = x.permute(1, 0, 2)                      # hw + 1, bs, c

    # Expand the transformer forward method.
    for layer in vlm.visual.transformer.resblocks[:-1]:
        x = layer(x)
    # Apply attention masks in the last layer.
    # x = torch.cat([
    #     x[:, i: i+1].repeat(len(m))
    #     for i, m in enumerate(masks)
    # ], dim=1)
    last = vlm.visual.transformer.resblocks[-1]
    x_ = last.ln_1(x)
    x = x + last.attn(x_, x_, x_, need_weights=False)[0]
    x = x + last.mlp(last.ln_2(x))

    x = x.permute(1, 0, 2)                      # bs, hw + 1, c

    x = vlm.visual.ln_post(x[:, 0, :])
    x = x @ vlm.visual.proj

    t = vlm.encode_text(targets)

    x = x / x.norm(dim=1, keepdim=True)
    t = t / t.norm(dim=1, keepdim=True)

    logit_scale = vlm.logit_scale.exp()
    logits_per_image = logit_scale * x @ t.t()
    return logits_per_image

if __name__ == "__main__":
    # vis_detr_attn_weights()
    # visualise_qpic_attn_weights()
    vis_attn_weights()
    # vis_all_attn_weights()
    # plot_pe_attn()
    # generate_box_binary_mask()
    # show_masks()

    # with open("hicodet/prompts.json", "r") as f:
    #     prompts = json.load(f)
    # targets = clip.tokenize(prompts).cuda()
    # model, preprocess = clip.load("ViT-B/32", device="cuda")
    # model.eval()
    # image = preprocess(Image.open("checkpoints/comp.png")).unsqueeze(0).cuda()
    # logits = test_clip(image, model, targets)
    # logits_, _ = model(image, targets)
    # print(torch.all(logits == logits_))
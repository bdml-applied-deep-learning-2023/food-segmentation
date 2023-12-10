import numpy as np
import matplotlib.pylab as plt
import torch
import clip
import argparse
import os

from models.dino_segmentation import DINOv2Segmenter
from utils.data import FoodRecognitionDataset
from utils.commons import save, load
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F


def train_step(context, data):
    model = context["model"]
    optimizer = context["optimizer"]

    image = data["image"]
    heatmap = data["heatmap"]
    mask = data["mask"]

    res = model(image, heatmap, mask)
    loss = res["loss"]
    acc_heatmap = res["acc_heatmap"]
    acc_existence = res["acc_existence"]

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "acc_heatmap": acc_heatmap.item(),
        "acc_existence": acc_existence.item()
    }


@torch.no_grad()
def eval_step(context, data):
    model = context["model"]

    image = data["image"]
    heatmap = data["heatmap"]
    mask = data["mask"]

    res = model(image, heatmap, mask)
    loss = res["loss"]
    acc_heatmap = res["acc_heatmap"]
    acc_existence = res["acc_existence"]

    return {
        "loss": loss.item(),
        "acc_heatmap": acc_heatmap.item(),
        "acc_existence": acc_existence.item()
    }


def train(opt):
    if opt.cpu is True:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{opt.gpu}")

    trainset = FoodRecognitionDataset("train", augmentation=True)
    validset = FoodRecognitionDataset("val")

    train_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=min(8, opt.batch_size))
    valid_loader = DataLoader(validset, batch_size=opt.batch_size, num_workers=min(8, opt.batch_size))

    model = DINOv2Segmenter(num_labels=124).to(device)
    optimizer = optim.AdamW([
        *model.classifier.parameters(),
        *model.exsistence_classifier.parameters()
    ], lr=opt.lr)

    context = {
        "model": model,
        "optimizer": optimizer,
        "best_val_loss": np.inf,
        "latest_ckpt_path": os.path.join(opt.checkpoint_dir, opt.name, "latest.pt"),
        "best_ckpt_path": os.path.join(opt.checkpoint_dir, opt.name, "best.pt"),
    }

    if opt.load is True:
        load(context, context["latest_ckpt_path"])

    train_loader_iter = iter(train_loader)

    train_loss = 0.0
    train_acc_heatmap = 0.0
    train_acc_existence = 0.0

    for itr in tqdm(range(opt.iters)):
        try:
            data_item = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            data_item = next(train_loader_iter)

        for key in data_item.keys():
            if hasattr(data_item[key], "to"):
                data_item[key] = data_item[key].to(device)

        res = train_step(context, data_item)
        train_loss += res["loss"] / opt.eval_iters
        train_acc_heatmap += res["acc_heatmap"] / opt.eval_iters
        train_acc_existence += res["acc_existence"] / opt.eval_iters
            
        if (itr + 1) % opt.eval_iters == 0:
            model.eval()
            val_loss = 0.0
            val_acc_heatmap = 0.0
            val_acc_existence = 0.0

            for data_item in tqdm(valid_loader, desc="Validation", leave=False):
                for key in data_item.keys():
                    if hasattr(data_item[key], "to"):
                        data_item[key] = data_item[key].to(device)

                res = eval_step(context, data_item)
                val_loss += res["loss"] / len(valid_loader)
                val_acc_heatmap += res["acc_heatmap"] / len(valid_loader)
                val_acc_existence += res["acc_existence"] / len(valid_loader)

            print(f"Train loss: {train_loss:.8f}, train heatmap acc: {train_acc_heatmap:.4f}, train existence acc: {train_acc_existence:.4f}")
            print(f"Valid loss: {val_loss:.8f}, valid heatmap acc: {val_acc_heatmap:.4f}, valid existence acc: {val_acc_existence:.4f}")

            if context["best_val_loss"] > val_loss:
                context["best_val_loss"] = val_loss
                save(context, context["best_ckpt_path"])
            save(context, context["latest_ckpt_path"])

            model.train()
            train_loss = 0.0
            train_acc_heatmap = 0.0
            train_acc_existence = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--eval_iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--name", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    train(opt)

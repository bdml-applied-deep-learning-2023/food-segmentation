import numpy as np
import cv2
import os
import json
import torch
import clip
import random
from copy import deepcopy
from PIL import Image
from pycocotools.coco import COCO
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T


class FoodRecognitionDataset(Dataset):

    DATA_DIR = "/data/food_recognition"
    TRAIN_DATA_DIR = "/data/food_recognition/train"
    VAL_DATA_DIR = "/data/food_recognition/val"
    ANNOT_FILE_NAME = "annotations.json"
    LABEL_MAP_PATH = "simple_label_map.json"
    PIXEL_MEAN = np.array([123.675, 116.28, 103.53])
    PIXEL_STD = np.array([58.395, 57.12, 57.375])
    IMAGE_SIZE = 448

    def __init__(self, split="train", augmentation=False):
        self.split = split
        self.augmentation = augmentation

        self._load_data_list(split)        
        self.cat_names = {
            cat_info["id"]: cat_info["name"]
            for cat_info in self.dataset.loadCats(self.dataset.getCatIds())
        }

    def __len__(self):
        return len(self.image_ids)
    
    def _load_data_list(self, split):
        if split == "train":
            annot_file_path = os.path.join(self.TRAIN_DATA_DIR, self.ANNOT_FILE_NAME)
        elif split == "val":
            annot_file_path = os.path.join(self.VAL_DATA_DIR, self.ANNOT_FILE_NAME)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.dataset = COCO(annot_file_path)

        self.label_map = self._load_label_map(self.LABEL_MAP_PATH)
        self.cat_ids = [item[1] for item in self.label_map.values()]
        self.image_ids = [
            self.dataset.getImgIds(catIds=[item[1]]) for item in self.label_map.values()
        ]
        self.image_ids = np.concatenate(self.image_ids)
        self.image_ids = np.unique(self.image_ids)
        self.image_ids = np.sort(self.image_ids)

        new_cat_names = list(map(lambda item: item[0], self.label_map.values()))
        new_cat_names = sorted(np.unique(new_cat_names))
        self.new_cat_names = {
            int(i): new_cat_names[i]
            for i in range(len(new_cat_names))
        }

        self.old_id_to_new_id = {
            int(value[1]): new_cat_names.index(value[0])
            for value in self.label_map.values()
        }

        self.new_cat_name_to_large_category = dict()
        for new_cat, old_id, large_cat in self.label_map.values():
            if new_cat not in self.new_cat_name_to_large_category.keys():
                self.new_cat_name_to_large_category[new_cat] = large_cat

        self.all_cat_text_tokens, self.all_cat_texts = self._get_all_cat_texts()

    def _load_label_map(self, label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        return label_map
    
    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image
    
    def _resize(self, image, *args):
        def __resize(image, standardize=False):
            H, W, C = image.shape
            if H > W:
                ratio = W/H
                image_placeholder = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, C), dtype=np.float32)
                image = cv2.resize(image, dsize=(int(self.IMAGE_SIZE*ratio), self.IMAGE_SIZE)).reshape(self.IMAGE_SIZE, int(self.IMAGE_SIZE*ratio), C)
                image_placeholder[:, :image.shape[1]] = image
                image = image_placeholder
            else:
                ratio = H/W
                image_placeholder = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, C), dtype=np.float32)
                image = cv2.resize(image, dsize=(self.IMAGE_SIZE, int(self.IMAGE_SIZE*ratio))).reshape(int(self.IMAGE_SIZE*ratio), self.IMAGE_SIZE, C)
                image_placeholder[:image.shape[0]] = image
                image = image_placeholder
            
            if standardize is True:
                image = (image - self.PIXEL_MEAN) / self.PIXEL_STD
            return image
        
        image = __resize(image, standardize=True)
        args = [__resize(arg, standardize=False) for arg in args]
        return image, *args
    
    def _preprocess_image(self, image, *args):
        if self.augmentation is True:
            image, *args = self._augmentation(image, *args)
        else:
            image = F.interpolate(image.unsqueeze(0), size=(self.IMAGE_SIZE, self.IMAGE_SIZE)).squeeze(0)
            args = [F.interpolate(arg.unsqueeze(0), size=(self.IMAGE_SIZE, self.IMAGE_SIZE)).squeeze(0) for arg in args]

        image = ((image * 255) - self.PIXEL_MEAN[:, None, None])/self.PIXEL_STD[:, None, None]
        image = image.float()
        return image, *args
    
    def _augmentation(self, image, *args):
        spatial_transform = Compose([
            RandomResizedCrop((self.IMAGE_SIZE, self.IMAGE_SIZE), scale=(0.8, 1.0)),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ])

        color_transform = T.Compose([
            T.ColorJitter(0.05, 0.05, 0.05, 0.05),
        ])

        image, *args = spatial_transform(image, *args)
        image = color_transform(image)

        return image, *args
    
    def _get_single_segmentation_heatmap(self, image, annot_info):
        _, H, W = image.shape

        segmentation_list = [np.array(list(map(int, segmentation))).reshape(-1, 2) for segmentation in annot_info["segmentation"]]
        heatmap = np.zeros((H, W), dtype=np.uint8)
        heatmap = cv2.fillPoly(heatmap, segmentation_list, 255)
        heatmap = heatmap.astype(np.float32) / 255
        heatmap[heatmap == 1] = 0.9
        heatmap[heatmap == 0] = 0.05
        return heatmap
    
    def _get_segmentation_heatmap(self, image, annot_info_list):
        _, H, W = image.shape
        heatmap = np.zeros((H, W, len(self.new_cat_names)), dtype=np.float32)

        for annot_info in annot_info_list:
            old_id = annot_info["category_id"]
            new_id = self.old_id_to_new_id[old_id]
            heatmap[..., new_id] = self._get_single_segmentation_heatmap(image, annot_info)

        heatmap = torch.tensor(heatmap, dtype=torch.float32).permute(2, 0, 1)
        return heatmap
    
    def _get_cat_mask(self, annot_info_list):
        mask = np.zeros((len(self.new_cat_names),), dtype=np.float32) + 0.01

        for annot_info in annot_info_list:
            old_id = annot_info["category_id"]
            new_id = self.old_id_to_new_id[old_id]
            mask[new_id] = 0.95

        return mask
    
    def _get_cat_text(self, cat_id):
        cat_info = self.dataset.loadCats([cat_id])[0]
        cat_name = cat_info["name"]
        refined_cat_name, _, large_category = self.label_map[cat_name]
        cat_text = f"A photo of {large_category} for {refined_cat_name}."
        return cat_text
    
    def _get_all_cat_texts(self):
        with open("cat_text.txt", "r") as f:
            cat_texts = f.readlines()

        cat_texts = list(map(lambda txt: txt.strip(), cat_texts))
        tokens = clip.tokenize(cat_texts)
        return tokens, cat_texts
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.dataset.loadImgs([image_id])[0]
        image_path = os.path.join(self.DATA_DIR, self.split, "images", image_info["file_name"])
        image = self._load_image(image_path)

        annot_ids = self.dataset.getAnnIds(imgIds=[image_id], catIds=self.cat_ids)
        annot_info_list = self.dataset.loadAnns(annot_ids)
        heatmap = self._get_segmentation_heatmap(image, annot_info_list)
        mask = self._get_cat_mask(annot_info_list)
        image, heatmap = self._preprocess_image(image, heatmap)

        return {
            "image": image,
            "all_cat_text_tokens": self.all_cat_text_tokens,
            "all_cat_texts": self.all_cat_texts,
            "heatmap": heatmap,
            "mask": mask
        }


class Compose(nn.Module):

    def __init__(self, transforms):
        super(Compose, self).__init__()

        self.transforms = transforms

    def forward(self, image, *args):
        for transform in self.transforms:
            image, *args = transform(image, *args)

        return image, *args


class RandomResizedCrop(nn.Module):

    def __init__(self, size, scale, ratio=[3/4, 4/3]):
        super(RandomResizedCrop, self).__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def forward(self, image, *args):
        i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = T.F.resized_crop(image, i, j, h, w, self.size)
        args = [T.F.resized_crop(arg, i, j, h, w, self.size) for arg in args]
        return image, *args


class RandomHorizontalFlip(nn.Module):

    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, image, *args):
        if random.random() < self.p:
            image = T.F.hflip(image)
            args = [T.F.hflip(arg) for arg in args]

        return image, *args
    
    
class RandomVerticalFlip(nn.Module):

    def __init__(self, p):
        super(RandomVerticalFlip, self).__init__()
        self.p = p

    def forward(self, image, *args):
        if random.random() < self.p:
            image = T.F.vflip(image)
            args = [T.F.vflip(arg) for arg in args]

        return image, *args

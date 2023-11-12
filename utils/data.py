import numpy as np
import cv2
import os
import torch
import random
from copy import deepcopy
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class FoodRecognitionDataset(Dataset):

    DATA_DIR = "/data/food_recognition"
    TRAIN_DATA_DIR = "/data/food_recognition/train"
    VAL_DATA_DIR = "/data/food_recognition/val"
    ANNOT_FILE_NAME = "annotations.json"
    PIXEL_MEAN = np.array([123.675, 116.28, 103.53])
    PIXEL_STD = np.array([58.395, 57.12, 57.375])
    IMAGE_SIZE = 256

    def __init__(self, split="train", augmentation=False):
        if split == "train":
            annot_file_path = os.path.join(self.TRAIN_DATA_DIR, self.ANNOT_FILE_NAME)
        elif split == "val":
            annot_file_path = os.path.join(self.VAL_DATA_DIR, self.ANNOT_FILE_NAME)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self.augmentation = augmentation
        
        self.dataset = COCO(annot_file_path)
        self.image_ids = self.dataset.getImgIds()
        self.cat_names = {
            cat_info["id"]: cat_info["name"]
            for cat_info in self.dataset.loadCats(self.dataset.getCatIds())
        }

    def __len__(self):
        return len(self.image_ids)
    
    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        image, *args = self._resize(image, *args)

        if self.augmentation is True:
            image, *args = self._augmentation(image, *args)

        return image, *args
    
    def _augmentation(self, image, *args):
        if random.random() < 0.5:
            image = deepcopy(image[:, ::-1])
            args = [deepcopy(arg[:, ::-1]) for arg in args]
        if random.random() < 0.5:
            distortion = np.random.rand(3) * 0.1 - 0.05
            image = image + distortion
        if random.random() < 0.5:
            image = image + np.random.normal(size=image.shape) * 0.05

        return image, *args
    
    def _get_segmentation_heatmap(self, image, annot_info, std=0.7):
        H, W, _ = image.shape

        segmentation_list = [np.array(list(map(int, segmentation))).reshape(-1, 2) for segmentation in annot_info["segmentation"]]
        mask = np.zeros((H, W), dtype=np.uint8)
        mask = cv2.fillPoly(mask, segmentation_list, 255)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        dist = dist / dist.max()
        heatmap = np.exp(-0.5 * ((dist - 1)/(std))**2)
        heatmap[dist == 0] = 0
        heatmap = heatmap / heatmap.max()
        return annot_info["category_id"], heatmap
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.dataset.loadImgs([image_id])[0]
        image_path = os.path.join(self.DATA_DIR, self.split, "images", image_info["file_name"])
        image = self._load_image(image_path)

        annot_ids = self.dataset.getAnnIds([image_id])
        annot_info_list = self.dataset.loadAnns(annot_ids)

        heatmap_res = list(map(lambda annot_info: self._get_segmentation_heatmap(image, annot_info), annot_info_list))
        cat_id_for_heatmap = list(map(lambda res: res[0], heatmap_res))
        heatmap = list(map(lambda res: res[1], heatmap_res))
        # cat_id_for_heamap, heatmap = list(zip(heatmap_res))
        cat_id_for_heatmap = np.array(cat_id_for_heatmap, dtype=np.int64)
        heatmap = np.stack(heatmap, axis=-1)

        cat_ids = list(map(lambda annot_info: annot_info["category_id"], annot_info_list))
        cat_names = list(map(lambda cat_id: self.cat_names[cat_id], cat_ids))
        random.shuffle(cat_names)
        cat_text = "A photo of a dish or drink for " + ", ".join(cat_names)
        
        image, heatmap = self._preprocess_image(image, heatmap)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).permute(2, 0, 1)
        cat_id_for_heatmap = torch.tensor(cat_id_for_heatmap, dtype=torch.long)

        return {
            "image": image,
            "text": cat_text,
            "heatmap": heatmap,
            "cat_id": cat_id_for_heatmap
        }

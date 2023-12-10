import torch
import numpy as np
import cv2
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from segment_anything import sam_model_registry, SamPredictor
from models.clicker import Clicker
from models.dino_segmentation import DINOv2Segmenter


class DINOwithSAM(nn.Module):

    # MODEL_TYPE = "vit_l"
    # CHECKPOINT = "./checkpoints/sam_vit_l_0b3195.pth"
    MODEL_TYPE = "vit_h"
    CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
    PIXEL_MEAN = np.array([123.675, 116.28, 103.53])
    PIXEL_STD = np.array([58.395, 57.12, 57.375])

    def __init__(self, dino_segmenter):
        super(DINOwithSAM, self).__init__()

        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT)
        self.predictor = SamPredictor(self.sam)
        
        self.dino_segmenter = dino_segmenter

    # @torch.no_grad()
    # def forward(self, image, heatmap, exist_pred, 
    #                   num_points=5, try_num_points=5
    #                   heatmap_threshold=0.85,
    #                   existence_threshold=0.85,
    #                   iou_threshold=0.7):
    #     """
    #     Arguments:
    #     ----------
    #     - image: (N, 3, H, W)
    #     - heatmap: (N, M, H, W)
    #     - exist_pred: (N, M)
    #     """

    #     N = image.size(0)
        
    #     resulting_masks = list(map(lambda i: self.forward_single(
    #         image[i], heatmap[i], exist_pred[i], 
    #         num_points, try_num_points,
    #         heatmap_threshold=heatmap_threshold,
    #         existence_threshold=existence_threshold,
    #         iou_threshold=iou_threshold
    #     ), range(N)))
    #     return resulting_masks

    # @torch.no_grad()
    # def forward_single(self, image, heatmap, exist_pred, 
    #                          num_points=5,
    #                          num_try_points=5,
    #                          heatmap_threshold=0.85,
    #                          existence_threshold=0.85,
    #                          iou_threshold=0.7,
    #                          difference_ratio_threshold=0.5):
    #     """
    #     Arguments:
    #     ----------
    #     - image: (3, H, W)
    #     - heatmap: (M, H, W)
    #     - exist_pred: (M,)
    #     """

    #     _, H, W = image.size()
    #     M = heatmap.size(0)

    #     image = image.permute(1, 2, 0).cpu().numpy()
    #     image = image * self.PIXEL_STD + self.PIXEL_MEAN
    #     image = image.astype(np.uint8)
    #     self.predictor.set_image(image)

    #     heatmap = heatmap.cpu().numpy()
    #     exist_pred = exist_pred.cpu().numpy()

    #     exist_index = np.arange(M)[exist_pred >= existence_threshold]
    #     if len(exist_index) == 0: return []

    #     heatmap = heatmap[exist_index]
    #     erosion_mask = self._get_erosion_mask(heatmap, heatmap_threshold)
    #     heatmap[erosion_mask == 0] = -99999
    #     heatmap = heatmap.reshape(-1, H*W)
        
    #     resulting_masks = []

    #     for m in range(heatmap.shape[0]):
    #         points = []
    #         labels = []
    #         positive_heatmap_m = deepcopy(heatmap[m])
    #         negative_heatmap_m = None

    #         resulting_mask = np.zeros((H, W), dtype=np.float32)
    #         label = 1

    #         for i in range(num_points):
    #             try_best_mask = None
    #             try_best_point = None
    #             try_best_label = None
    #             try_mask_score = -999999

    #             for _ in range(num_try_points):
    #                 try_mask = deepcopy(resulting_mask)

    #                 if label == 1:
    #                     positive_heatmap_m = positive_heatmap_m - positive_heatmap_m.max()
    #                     exp = np.exp(positive_heatmap_m)
    #                     p = exp / exp.sum()
    #                     loc = np.random.choice(np.arange(H*W), p=p)

    #                     h = loc // W
    #                     w = loc % W

    #                     points.append([w, h])
    #                     labels.append(label)

    #                     points_arr = points + [w, h]
    #                     labels_arr = labels + label

    #                     print("Point coords:", points_arr)
    #                     print("Point labels:", labels_arr)

    #                     masks, scores, logits = self.predictor.predict(
    #                         point_coords=points_arr,
    #                         point_labels=labels_arr,
    #                         multimask_output=True
    #                     )

    #                     pred_mask = masks[0]
    #                     try_mask += pred_mask
    #                     try_mask[try_mask > 1] = 1

    #                     try_score = self._compute_mask_score(try_mask, p.reshape(H, W))  
    #                     print("Try score:", try_score)

    #                     if try_score > try_mask_score:
    #                         try_mask_score = try_score
    #                         try_best_mask = try_mask
    #                         try_best_point = [w, h]
    #                         try_best_label = label

    #             points.append(try_best_point)
    #             labels.append(try_best_label)

    #             binary_heatmap = (heatmap[m].reshape(H, W) >= heatmap_threshold).astype(np.float32)
    #             coverage_ratio = self._compute_coverage_ratio(try_best_mask, binary_heatmap)
    #             if coverage_ratio > 0.8:
    #                 break

    #             positive_heatmap_m[resulting_mask.reshape(-1) == 1] = -99999

    #             # difference_ratio, difference_heatmap = self._compute_difference_ratio(binary_heatmap, resulting_mask)
    #             # print("difference_ratio:", difference_ratio)
    #             # if difference_ratio > difference_ratio_threshold:
    #             #     label = 0
    #             #     negative_heatmap_m = difference_heatmap
    #             # else:
    #             #     label = 1

    #         resulting_masks.append(resulting_mask)

    #     return resulting_masks

    def _get_erosion_mask(self, heatmap, heatmap_threshold=0.5, iterations=3):
        M, H, W = heatmap.shape

        masks = list(map(lambda m: self._get_single_erosion_mask(heatmap[m], heatmap_threshold, iterations), range(M)))
        masks = np.stack(masks, axis=0)
        return masks
    
    def _get_single_erosion_mask(self, heatmap, heatmap_threshold, iterations):
        """
        Arguments:
        ----------
        - heatmap: (H, W)
        """

        binary_heatmap = (heatmap >= heatmap_threshold).astype(np.float32)
        mask = cv2.erode(binary_heatmap, np.ones((3, 3), dtype=np.float32), iterations=iterations)
        return mask
    
    def _compute_iou(self, binary_heatmap, mask):
        """
        Arguments:
        ----------
        - binary_heatmap: (H, W)
        - mask: (H, W)
        - heatmap_threshold: float
        """

        binary_heatmap = binary_heatmap.reshape(-1)
        mask = mask.reshape(-1)

        intersection = ((mask == 1) & (binary_heatmap == 1)).astype(np.float32).sum()
        union = ((mask == 1) | (binary_heatmap == 1)).astype(np.float32).sum()

        iou = intersection / (union + 1e-6)
        return iou
    
    def _compute_difference_ratio(self, mask, binary_heatmap):
        """
        Arguments:
        ----------
        - mask: (H, W)
        - binary_heatmap: (H, W)
        """

        # difference_size = ((binary_heatmap == 0) & (mask == 1)).astype(np.float32).sum() \
        #                 + ((binary_heatmap == 1) & (mask == 0)).astype(np.float32).sum()
        # total_size = (binary_heatmap == 1).astype(np.float32).sum()

        difference_size = ((binary_heatmap == 0) & (mask == 1)).astype(np.float32).sum()
        total_size = (mask == 1).astype(np.float32).sum()

        ratio = difference_size / total_size
        return ratio
    
    def _compute_mask_score(self, mask, heatmap):
        mask_score = (mask * heatmap).sum()
        size_penalty = mask.sum()

        return mask_score - 0.01*size_penalty
    
    def _compute_coverage_ratio(self, mask, binary_heatmap):
        covered = (mask * binary_heatmap).sum()
        total = (binary_heatmap == 1).astype(np.float32).sum()
        return covered / total

    @torch.no_grad()
    def forward(self, image, 
                      num_points=5, 
                      heatmap_threshold=0.85,
                      existence_threshold=0.85,
                      coverage_ratio_threshold=0.8,
                      difference_ratio_threshold=0.8):
        """
        Arguments:
        ----------
        - image: (N, 3, H, W)
        """

        N = image.size(0)

        res = self.dino_segmenter(image)
        exist_pred = torch.sigmoid(res["logits_existence"])
        heatmap_pred = torch.sigmoid(res["logits_heatmap"])
        
        resulting_masks = list(map(lambda i: self.forward_single(
            image[i], heatmap_pred[i], exist_pred[i], num_points,
            heatmap_threshold=heatmap_threshold,
            existence_threshold=existence_threshold,
            coverage_ratio_threshold=coverage_ratio_threshold,
            difference_ratio_threshold=difference_ratio_threshold
        ), range(N)))
        
        return {
            "resulting_masks": resulting_masks,
            "exist_pred": exist_pred,
            "heatmap_pred": heatmap_pred
        }

    @torch.no_grad()
    def forward_single(self, image, heatmap, exist_pred, 
                             num_points=5,
                             heatmap_threshold=0.85,
                             existence_threshold=0.85,
                             coverage_ratio_threshold=0.8,
                             difference_ratio_threshold=0.8):
        """
        Arguments:
        ----------
        - image: (3, H, W)
        - heatmap: (M, H, W)
        - exist_pred: (M,)
        """

        _, H, W = image.size()
        M = heatmap.size(0)

        image = image.permute(1, 2, 0).cpu().numpy()
        image = image * self.PIXEL_STD + self.PIXEL_MEAN
        image = image.astype(np.uint8)
        self.predictor.set_image(image)

        heatmap = heatmap.cpu().numpy()
        exist_pred = exist_pred.cpu().numpy()

        exist_index = np.arange(M)[exist_pred >= existence_threshold]
        if len(exist_index) == 0: return []

        heatmap = heatmap[exist_index]
        heatmap = (heatmap > heatmap_threshold).astype(np.uint8) * 255
        heatmap = [
            cv2.distanceTransform(heatmap[i], cv2.DIST_L2, 3).astype(np.float32)
            for i in range(len(exist_index))
        ]
        heatmap = np.stack(heatmap, axis=0)
        
        resulting_masks = []

        for m in range(heatmap.shape[0]):
            points = []
            labels = []
            positive_heatmap_m = heatmap[m]
            negative_heatmap_m = None
            label = 1

            for i in range(num_points):
                if label == 1:
                    # positive_logits = positive_heatmap_m - positive_heatmap_m.max()
                    # exp = np.exp(positive_logits.reshape(-1))
                    # p = exp / exp.sum()
                    # if len(p.shape) != 1: print(p.shape)
                    # loc = np.random.choice(np.arange(H*W), p=p)

                    loc = positive_heatmap_m.argmax()

                    # logits = positive_heatmap_m - positive_heatmap_m.max()
                    # exp = np.exp(logits)
                    # p = exp / exp.sum()
                    # loc = np.random.choice(np.arange(H*W), p=p)
                else:
                    # negative_logits = negative_heatmap_m - negative_heatmap_m.max()
                    # exp = np.exp(negative_logits.reshape(-1))
                    # p = exp / exp.sum()
                    # if len(p.shape) != 1: print(p.shape)
                    # loc = np.random.choice(np.arange(H*W), p=p)

                    loc = negative_heatmap_m.argmax()

                    # logits = negative_heatmap_m - negative_heatmap_m.max()
                    # exp = np.exp(logits)
                    # p = exp / exp.sum()
                    # loc = np.random.choice(np.arange(H*W), p=p)

                h = loc // W
                w = loc % W

                points.append([w, h])
                labels.append(label)

                points_arr = np.array(points)
                labels_arr = np.array(labels)

                # print("Point coords:", points_arr)
                # print("Point labels:", labels_arr)

                masks, scores, logits = self.predictor.predict(
                    point_coords=points_arr,
                    point_labels=labels_arr,
                    multimask_output=True
                )

                binary_heatmap = (heatmap[m] > 0).astype(np.float32)

                best_mask = None
                best_score = 0.0

                best_coverage_ratio = 0.0
                best_difference_ratio = 0.0

                for l in range(len(masks)):
                    pred_mask = masks[l]
                    pred_mask = pred_mask.astype(np.float32)

                    coverage_ratio = self._compute_coverage_ratio(pred_mask, binary_heatmap)
                    difference_ratio = self._compute_difference_ratio(pred_mask, binary_heatmap)

                    if coverage_ratio / (difference_ratio + 1e-6) > best_score:
                        best_mask = pred_mask
                        best_score = coverage_ratio / (difference_ratio + 1e-6)
                        best_coverage_ratio = coverage_ratio
                        best_difference_ratio = difference_ratio

                resulting_mask = best_mask
                # print("coverage ratio:", best_coverage_ratio)
                # print("difference ratio:", best_difference_ratio)

                if best_coverage_ratio >= coverage_ratio_threshold and best_difference_ratio <= difference_ratio_threshold:
                    break
                elif best_difference_ratio > difference_ratio_threshold:
                    negative_heatmap_m = (resulting_mask * (1 - binary_heatmap) * 255).astype(np.uint8)
                    negative_heatmap_m = cv2.distanceTransform(negative_heatmap_m, cv2.DIST_L2, 3).astype(np.float32)
                    label = 0
                    continue
                elif best_coverage_ratio < coverage_ratio_threshold:
                    positive_heatmap_m = (binary_heatmap * (1 - resulting_mask) * 255).astype(np.uint8)
                    positive_heatmap_m = cv2.distanceTransform(positive_heatmap_m, cv2.DIST_L2, 3).astype(np.float32)
                    label = 1
                    continue
                else:
                    raise ValueError("This condition is unreachable!")

            resulting_masks.append(resulting_mask)

        return resulting_masks

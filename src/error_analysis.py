#!/usr/bin/env python3
"""
Error Analysis with IoU-Based Element Matching
This script runs models from scratch and uses IoU to correctly match GT and predicted elements.
"""

import sys
import os
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from pycocotools import mask as mask_util
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image

# Add Mask2Former path
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
sys.path.insert(0, MASK2FORMER_ROOT)

from mask2former import add_maskformer2_config

# Import dataset registration modules
import datasets_semantic
import datasets

# Setup logger
setup_logger()
logger = logging.getLogger("error_analysis")

# Corrosion colors and classes
CORROSION_COLORS = {
    "Background": (0, 0, 0),
    "Fair": (0, 255, 0),
    "Poor": (255, 255, 0),
    "Severe": (255, 0, 0)
}

CORROSION_CLASS_IDS = {
    "Background": 0,
    "Fair": 1,
    "Poor": 2,
    "Severe": 3
}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def match_instances_by_iou(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match GT and predicted instances using IoU with Hungarian algorithm.
    
    Returns:
        matches: List of (gt_idx, pred_idx) pairs
        unmatched_gt: List of GT indices with no match
        unmatched_pred: List of pred indices with no match
    """
    if not gt_masks or not pred_masks:
        return [], list(range(len(gt_masks))), list(range(len(pred_masks)))
    
    # Calculate IoU matrix
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = calculate_iou(gt_mask, pred_mask)
    
    # Use Hungarian algorithm for optimal matching
    # Maximize IoU = minimize negative IoU
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    
    # Filter matches by IoU threshold
    matches = []
    matched_gt = set()
    matched_pred = set()
    
    for gt_idx, pred_idx in zip(row_indices, col_indices):
        if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
            matches.append((gt_idx, pred_idx))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
    
    # Find unmatched instances
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    
    return matches, unmatched_gt, unmatched_pred


class ErrorAnalyzer:
    """Error analysis with IoU-based instance matching."""
    
    def __init__(
        self,
        semantic_cfg,
        instance_cfg,
        instance_gt_json: str,
        semantic_gt_dir: str,
        images_dir: str,
        output_dir: str,
        dataset_name: str,
        instance_dataset_name: str,
        iou_threshold: float = 0.5,
        min_confidence: float = 0.5
    ):
        self.semantic_cfg = semantic_cfg
        self.instance_cfg = instance_cfg
        self.instance_gt_json = instance_gt_json
        self.semantic_gt_dir = semantic_gt_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.instance_dataset_name = instance_dataset_name
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        
        # Create output directories
        self.data_dir = os.path.join(output_dir, "data")
        self.vis_dir = os.path.join(output_dir, "visualizations")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.error_dir = os.path.join(output_dir, "error_analysis")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        
        # Build models
        logger.info("Building semantic segmentation model...")
        self.semantic_model = build_model(semantic_cfg)
        self.semantic_model.eval()
        DetectionCheckpointer(self.semantic_model).load(semantic_cfg.MODEL.WEIGHTS)
        
        logger.info("Building instance segmentation model...")
        self.instance_model = build_model(instance_cfg)
        self.instance_model.eval()
        DetectionCheckpointer(self.instance_model).load(instance_cfg.MODEL.WEIGHTS)
        
        # Load GT data
        logger.info(f"Loading instance ground truth from {instance_gt_json}")
        with open(instance_gt_json, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        logger.info(f"Loaded {len(self.image_id_to_info)} images with {len(self.coco_data['annotations'])} GT annotations")
        
        # Get class names
        instance_metadata = MetadataCatalog.get(self.instance_dataset_name)
        self.structural_classes = instance_metadata.thing_classes if hasattr(instance_metadata, "thing_classes") else []
        
        semantic_metadata = MetadataCatalog.get(self.dataset_name)
        self.corrosion_classes = semantic_metadata.stuff_classes if hasattr(semantic_metadata, "stuff_classes") else ["Background", "Fair", "Poor", "Severe"]
        
        logger.info(f"Structural classes: {self.structural_classes}")
        logger.info(f"Corrosion classes: {self.corrosion_classes}")
        
        # Results storage
        self.results = []
        self.matching_stats = {
            "total_gt_instances": 0,
            "total_pred_instances": 0,
            "matched_instances": 0,
            "unmatched_gt": 0,
            "unmatched_pred": 0,
            "false_negatives": [],  # GT elements with no prediction
            "false_positives": []   # Predictions with no GT
        }
        
        self.error_metrics = {
            "per_element": [],
            "per_corrosion_type": defaultdict(list),
            "overall": {}
        }
    
    def load_semantic_gt_mask(self, image_filename: str) -> Optional[np.ndarray]:
        """Load semantic segmentation ground truth mask."""
        base_name = Path(image_filename).stem
        possible_names = [
            f"{base_name}.png",
            f"{base_name}_gtFine_labelIds.png",
            f"{base_name}_labelIds.png"
        ]
        
        for name in possible_names:
            mask_path = os.path.join(self.semantic_gt_dir, name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask
        
        logger.warning(f"Semantic GT mask not found for {image_filename}")
        return None
    
    def decode_rle_mask(self, segmentation, height: int, width: int) -> Optional[np.ndarray]:
        """Decode RLE or polygon segmentation to binary mask."""
        if isinstance(segmentation, dict):
            if 'counts' in segmentation:
                rle = segmentation
            else:
                return None
            mask = mask_util.decode(rle)
        elif isinstance(segmentation, list):
            if len(segmentation) == 0:
                return np.zeros((height, width), dtype=np.uint8)
            rles = mask_util.frPyObjects(segmentation, height, width)
            rle = mask_util.merge(rles)
            mask = mask_util.decode(rle)
        else:
            return None
        
        return mask.astype(bool)
    
    def run_semantic_prediction(self, image: np.ndarray) -> np.ndarray:
        """Run semantic segmentation model."""
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image_tensor, "height": height, "width": width}
        
        with torch.no_grad():
            outputs = self.semantic_model([inputs])[0]
        
        if "sem_seg" in outputs:
            return outputs["sem_seg"].argmax(dim=0).cpu().numpy()
        else:
            return np.zeros((height, width), dtype=np.int64)
    
    def run_instance_prediction(self, image: np.ndarray):
        """Run instance segmentation model."""
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image_tensor, "height": height, "width": width}
        
        with torch.no_grad():
            outputs = self.instance_model([inputs])[0]
        
        return outputs.get("instances", None)
    
    def analyze_image(self, image_id: int, generate_visualization: bool = True) -> Optional[Dict]:
        """Analyze one image with IoU-based matching."""
        if image_id not in self.image_id_to_info:
            return None
        
        image_info = self.image_id_to_info[image_id]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.images_dir, image_filename)
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        # Get GT data
        semantic_gt = self.load_semantic_gt_mask(image_filename)
        if semantic_gt is None:
            return None
        
        if semantic_gt.shape != (height, width):
            semantic_gt = cv2.resize(semantic_gt, (width, height), interpolation=cv2.INTER_NEAREST)
        
        gt_annotations = self.annotations_by_image[image_id]
        if not gt_annotations:
            return None
        
        # Decode GT instance masks
        gt_masks = []
        gt_classes = []
        gt_ann_ids = []
        
        for ann in gt_annotations:
            mask = self.decode_rle_mask(ann.get('segmentation'), height, width)
            if mask is not None and mask.any():
                gt_masks.append(mask)
                gt_classes.append(self.category_id_to_name.get(ann['category_id'], "Unknown"))
                gt_ann_ids.append(ann['id'])
        
        # Run predictions
        semantic_pred = self.run_semantic_prediction(image_rgb)
        instance_pred = self.run_instance_prediction(image_rgb)
        
        # Extract predicted instance masks
        pred_masks = []
        pred_classes = []
        pred_scores = []
        
        if instance_pred is not None and len(instance_pred) > 0:
            if hasattr(instance_pred, "pred_masks"):
                for i in range(len(instance_pred)):
                    score = instance_pred.scores[i].item() if hasattr(instance_pred, "scores") else 1.0
                    if score >= self.min_confidence:
                        mask = instance_pred.pred_masks[i].cpu().numpy()
                        class_id = instance_pred.pred_classes[i].item() if hasattr(instance_pred, "pred_classes") else 0
                        class_name = self.structural_classes[class_id] if class_id < len(self.structural_classes) else "Unknown"
                        
                        pred_masks.append(mask)
                        pred_classes.append(class_name)
                        pred_scores.append(score)
        
        # Update stats
        self.matching_stats["total_gt_instances"] += len(gt_masks)
        self.matching_stats["total_pred_instances"] += len(pred_masks)
        
        # Match instances using IoU
        matches, unmatched_gt, unmatched_pred = match_instances_by_iou(
            gt_masks, pred_masks, self.iou_threshold
        )
        
        self.matching_stats["matched_instances"] += len(matches)
        self.matching_stats["unmatched_gt"] += len(unmatched_gt)
        self.matching_stats["unmatched_pred"] += len(unmatched_pred)
        
        # Record unmatched instances
        for gt_idx in unmatched_gt:
            self.matching_stats["false_negatives"].append({
                "image": image_filename,
                "class": gt_classes[gt_idx],
                "annotation_id": gt_ann_ids[gt_idx]
            })
        
        for pred_idx in unmatched_pred:
            self.matching_stats["false_positives"].append({
                "image": image_filename,
                "class": pred_classes[pred_idx],
                "score": pred_scores[pred_idx]
            })
        
        # Analyze matched pairs
        matched_elements = []
        
        for gt_idx, pred_idx in matches:
            gt_mask = gt_masks[gt_idx]
            pred_mask = pred_masks[pred_idx]
            class_name = gt_classes[gt_idx]
            
            iou = calculate_iou(gt_mask, pred_mask)
            
            gt_area = np.sum(gt_mask)
            pred_area = np.sum(pred_mask)
            
            # Calculate GT corrosion within GT element
            gt_corrosion_areas = {}
            for corr_name, corr_id in CORROSION_CLASS_IDS.items():
                if corr_name == "Background":
                    continue
                corr_mask = (semantic_gt == corr_id)
                intersection = np.logical_and(gt_mask, corr_mask)
                area = np.sum(intersection)
                ratio = area / gt_area if gt_area > 0 else 0.0
                gt_corrosion_areas[corr_name] = {
                    "area_pixels": int(area),
                    "area_ratio": float(ratio)
                }
            
            # Calculate Pred corrosion within Pred element
            pred_corrosion_areas = {}
            for corr_name, corr_id in CORROSION_CLASS_IDS.items():
                if corr_name == "Background":
                    continue
                corr_mask = (semantic_pred == corr_id)
                intersection = np.logical_and(pred_mask, corr_mask)
                area = np.sum(intersection)
                ratio = area / pred_area if pred_area > 0 else 0.0
                pred_corrosion_areas[corr_name] = {
                    "area_pixels": int(area),
                    "area_ratio": float(ratio)
                }
            
            # Calculate errors
            element_errors = {
                "image_file": image_filename,
                "element_type": class_name,
                "gt_annotation_id": gt_ann_ids[gt_idx],
                "pred_score": pred_scores[pred_idx],
                "iou": float(iou),
                "gt_area_px": int(gt_area),
                "pred_area_px": int(pred_area)
            }
            
            for corr_type in ["Fair", "Poor", "Severe"]:
                gt_ratio = gt_corrosion_areas[corr_type]["area_ratio"]
                pred_ratio = pred_corrosion_areas[corr_type]["area_ratio"]
                
                abs_error = abs(pred_ratio - gt_ratio)
                rel_error = (abs_error / gt_ratio * 100) if gt_ratio > 0.001 else 0.0
                
                element_errors[f"{corr_type}_GT_ratio_%"] = gt_ratio * 100
                element_errors[f"{corr_type}_Pred_ratio_%"] = pred_ratio * 100
                element_errors[f"{corr_type}_Abs_error_%"] = abs_error * 100
                element_errors[f"{corr_type}_Rel_error_%"] = rel_error
                
                self.error_metrics["per_corrosion_type"][corr_type].append({
                    "gt": gt_ratio,
                    "pred": pred_ratio,
                    "abs_error": abs_error,
                    "rel_error": rel_error
                })
            
            self.error_metrics["per_element"].append(element_errors)
            
            matched_elements.append({
                "gt_annotation_id": gt_ann_ids[gt_idx],
                "class_name": class_name,
                "iou": float(iou),
                "gt_area": int(gt_area),
                "pred_area": int(pred_area),
                "gt_corrosion": gt_corrosion_areas,
                "pred_corrosion": pred_corrosion_areas
            })
        
        # Generate visualizations
        if generate_visualization and matched_elements:
            # Create overview comparison
            self._visualize_comparison(
                image, semantic_gt, semantic_pred,
                gt_masks, pred_masks, matches,
                unmatched_gt, unmatched_pred,
                gt_classes, pred_classes,
                image_filename
            )
            
            # Create element-wise corrosion comparison
            self._visualize_elementwise_corrosion(
                image, semantic_gt, semantic_pred,
                gt_masks, pred_masks, matches,
                gt_classes, pred_classes,
                matched_elements,
                image_filename
            )
        
        return {
            "image_id": image_id,
            "file_name": image_filename,
            "matched_elements": matched_elements,
            "num_gt": len(gt_masks),
            "num_pred": len(pred_masks),
            "num_matched": len(matches),
            "num_unmatched_gt": len(unmatched_gt),
            "num_unmatched_pred": len(unmatched_pred)
        }
    
    def _visualize_comparison(
        self, image, semantic_gt, semantic_pred,
        gt_masks, pred_masks, matches,
        unmatched_gt, unmatched_pred,
        gt_classes, pred_classes,
        filename
    ):
        """Generate comprehensive comparison visualization."""
        h, w = image.shape[:2]
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. GT Semantic
        gt_sem_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for corr_name, corr_id in CORROSION_CLASS_IDS.items():
            if corr_name != "Background":
                mask = (semantic_gt == corr_id)
                color = CORROSION_COLORS[corr_name]
                gt_sem_vis[mask] = color
        
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].imshow(gt_sem_vis, alpha=0.5)
        axes[0, 0].set_title(f'GT Semantic Segmentation', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Pred Semantic
        pred_sem_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for corr_name, corr_id in CORROSION_CLASS_IDS.items():
            if corr_name != "Background":
                mask = (semantic_pred == corr_id)
                color = CORROSION_COLORS[corr_name]
                pred_sem_vis[mask] = color
        
        axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 1].imshow(pred_sem_vis, alpha=0.5)
        axes[0, 1].set_title(f'Predicted Semantic Segmentation', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. GT Instances with matches
        gt_inst_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        for gt_idx, mask in enumerate(gt_masks):
            color = (0, 255, 0) if gt_idx not in unmatched_gt else (255, 0, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_inst_vis, contours, -1, color, 2)
        
        axes[1, 0].imshow(gt_inst_vis)
        axes[1, 0].set_title(f'GT Instances (Green=Matched, Red=Unmatched)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Pred Instances with matches
        pred_inst_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        for pred_idx, mask in enumerate(pred_masks):
            color = (0, 255, 0) if pred_idx not in unmatched_pred else (255, 165, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_inst_vis, contours, -1, color, 2)
        
        axes[1, 1].imshow(pred_inst_vis)
        axes[1, 1].set_title(f'Pred Instances (Green=Matched, Orange=Unmatched)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(self.vis_dir, f"overview_{Path(filename).stem}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_elementwise_corrosion(
        self, image, semantic_gt, semantic_pred,
        gt_masks, pred_masks, matches,
        gt_classes, pred_classes,
        matched_elements,
        filename
    ):
        """Generate side-by-side element-wise corrosion visualization with original image."""
        h, w = image.shape[:2]
        
        # Create side-by-side comparison
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: GT element-wise corrosion
        gt_elementwise = image.copy()
        
        for gt_idx, pred_idx in matches:
            gt_mask = gt_masks[gt_idx]
            
            # Draw structural element outline (white, thicker)
            contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_elementwise, contours, -1, (255, 255, 255), 3)
            
            # Add element label
            y_positions = np.where(gt_mask)[0]
            x_positions = np.where(gt_mask)[1]
            if len(y_positions) > 0:
                text_x = np.min(x_positions)
                text_y = np.min(y_positions) - 10
                if text_y < 10:
                    text_y = np.min(y_positions) + 20
                cv2.putText(gt_elementwise, gt_classes[gt_idx], (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Overlay corrosion inside element with transparency
            for corr_name, corr_id in CORROSION_CLASS_IDS.items():
                if corr_name == "Background":
                    continue
                
                # Get corrosion mask
                corr_mask = (semantic_gt == corr_id)
                
                # Intersection with element
                intersection = np.logical_and(gt_mask, corr_mask)
                
                if np.any(intersection):
                    # Create overlay with original color, just add transparency
                    color_bgr = CORROSION_COLORS[corr_name][::-1]  # RGB to BGR
                    overlay = gt_elementwise.copy()
                    overlay[intersection] = color_bgr
                    # Blend with original image (alpha=0.5 for 50% transparency)
                    cv2.addWeighted(gt_elementwise, 0.6, overlay, 0.4, 0, gt_elementwise)
        
        # Right side: Predicted element-wise corrosion
        pred_elementwise = image.copy()
        
        for gt_idx, pred_idx in matches:
            pred_mask = pred_masks[pred_idx]
            
            # Draw structural element outline (white, thicker)
            contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_elementwise, contours, -1, (255, 255, 255), 3)
            
            # Add element label
            y_positions = np.where(pred_mask)[0]
            x_positions = np.where(pred_mask)[1]
            if len(y_positions) > 0:
                text_x = np.min(x_positions)
                text_y = np.min(y_positions) - 10
                if text_y < 10:
                    text_y = np.min(y_positions) + 20
                cv2.putText(pred_elementwise, pred_classes[pred_idx], (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Overlay corrosion inside element with transparency
            for corr_name, corr_id in CORROSION_CLASS_IDS.items():
                if corr_name == "Background":
                    continue
                
                # Get corrosion mask
                corr_mask = (semantic_pred == corr_id)
                
                # Intersection with element
                intersection = np.logical_and(pred_mask, corr_mask)
                
                if np.any(intersection):
                    # Create overlay with original color, just add transparency
                    color_bgr = CORROSION_COLORS[corr_name][::-1]  # RGB to BGR
                    overlay = pred_elementwise.copy()
                    overlay[intersection] = color_bgr
                    # Blend with original image (alpha=0.5 for 50% transparency)
                    cv2.addWeighted(pred_elementwise, 0.6, overlay, 0.4, 0, pred_elementwise)
        
        # Combine left and right
        comparison[:, :w] = gt_elementwise
        comparison[:, w:] = pred_elementwise
        
        # Add dividing line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
        
        # Add labels at top
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Ground Truth Element-Wise Corrosion", (10, 30),
                   font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "Predicted Element-Wise Corrosion", (w + 10, 30),
                   font, 0.8, (255, 255, 255), 2)
        
        # Create color legend
        legend_height = 30 * 3  # Fair, Poor, Severe
        legend = np.ones((legend_height, w * 2, 3), dtype=np.uint8) * 255
        
        for idx, corr_name in enumerate(["Fair", "Poor", "Severe"]):
            color_bgr = CORROSION_COLORS[corr_name][::-1]  # RGB to BGR
            cv2.rectangle(legend, (10, idx * 30 + 5), (30, idx * 30 + 25), color_bgr, -1)
            cv2.rectangle(legend, (10, idx * 30 + 5), (30, idx * 30 + 25), (0, 0, 0), 1)
            cv2.putText(legend, f"{corr_name} Corrosion", (40, idx * 30 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add legend to comparison
        comparison_with_legend = np.vstack([comparison, legend])
        
        # Save
        vis_path = os.path.join(self.vis_dir, f"elementwise_{Path(filename).stem}.png")
        cv2.imwrite(vis_path, comparison_with_legend)
    
    def analyze_all_images(self, generate_visualizations: bool = True):
        """Analyze all images with matching."""
        logger.info(f"Analyzing {len(self.image_id_to_info)} images for {self.dataset_name}...")
        
        for image_id in tqdm(self.image_id_to_info.keys(), desc="Processing images"):
            result = self.analyze_image(image_id, generate_visualizations)
            if result:
                self.results.append(result)
        
        logger.info(f"Successfully analyzed {len(self.results)} images")
        self._calculate_overall_statistics()
    
    def _calculate_overall_statistics(self):
        """Calculate overall error statistics."""
        logger.info("Calculating overall error statistics...")
        
        overall = {}
        
        for corr_type in ["Fair", "Poor", "Severe"]:
            if corr_type not in self.error_metrics["per_corrosion_type"]:
                continue
            
            errors = self.error_metrics["per_corrosion_type"][corr_type]
            abs_errors = [e["abs_error"] for e in errors]
            rel_errors = [e["rel_error"] for e in errors if e["rel_error"] > 0]
            
            overall[corr_type] = {
                "mean_abs_error_%": float(np.mean(abs_errors) * 100) if abs_errors else 0.0,
                "std_abs_error_%": float(np.std(abs_errors) * 100) if abs_errors else 0.0,
                "max_abs_error_%": float(np.max(abs_errors) * 100) if abs_errors else 0.0,
                "mean_rel_error_%": float(np.mean(rel_errors)) if rel_errors else 0.0,
                "std_rel_error_%": float(np.std(rel_errors)) if rel_errors else 0.0,
                "num_comparisons": len(errors)
            }
        
        self.error_metrics["overall"] = overall
        
        # Log matching statistics
        logger.info("=" * 60)
        logger.info(f"INSTANCE MATCHING STATISTICS ({self.dataset_name})")
        logger.info("=" * 60)
        logger.info(f"Total GT instances: {self.matching_stats['total_gt_instances']}")
        logger.info(f"Total Pred instances: {self.matching_stats['total_pred_instances']}")
        logger.info(f"Matched (IoU ≥ {self.iou_threshold}): {self.matching_stats['matched_instances']}")
        logger.info(f"Unmatched GT (False Negatives): {self.matching_stats['unmatched_gt']}")
        logger.info(f"Unmatched Pred (False Positives): {self.matching_stats['unmatched_pred']}")
        logger.info(f"Recall: {self.matching_stats['matched_instances'] / max(1, self.matching_stats['total_gt_instances']) * 100:.2f}%")
        logger.info(f"Precision: {self.matching_stats['matched_instances'] / max(1, self.matching_stats['total_pred_instances']) * 100:.2f}%")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"CORROSION PREDICTION ERRORS ({self.dataset_name} - matched instances)")
        logger.info("=" * 60)
        for corr_type, stats in overall.items():
            logger.info(f"\n{corr_type}:")
            logger.info(f"  Mean Absolute Error: {stats['mean_abs_error_%']:.2f}%")
            logger.info(f"  Std Absolute Error: {stats['std_abs_error_%']:.2f}%")
            logger.info(f"  Max Absolute Error: {stats['max_abs_error_%']:.2f}%")
            logger.info(f"  Mean Relative Error: {stats['mean_rel_error_%']:.2f}%")
    
    def save_results(self, output_format: str = "all"):
        """Save all results."""
        logger.info("Saving results...")
        
        # Save matching statistics
        matching_path = os.path.join(self.data_dir, "matching_statistics.json")
        with open(matching_path, 'w') as f:
            json.dump(self.matching_stats, f, indent=2)
        
        # Save error metrics
        if output_format in ["json", "all"]:
            error_path = os.path.join(self.error_dir, "error_metrics.json")
            with open(error_path, 'w') as f:
                json.dump({
                    "per_element_errors": self.error_metrics["per_element"],
                    "overall_statistics": self.error_metrics["overall"]
                }, f, indent=2)
        
        # Save as CSV
        if output_format in ["csv", "all"] and self.error_metrics["per_element"]:
            csv_path = os.path.join(self.error_dir, "element_level_errors.csv")
            df = pd.DataFrame(self.error_metrics["per_element"])
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV to {csv_path}")
        
        # Save as Excel
        if output_format in ["excel", "all"]:
            self._save_excel_report()
        
        # Generate plots
        self._generate_plots()
        
        # Generate report
        self._generate_report()
    
    def _save_excel_report(self):
        """Save comprehensive Excel report."""
        excel_path = os.path.join(self.error_dir, "error_analysis.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Element-level errors
            if self.error_metrics["per_element"]:
                df = pd.DataFrame(self.error_metrics["per_element"])
                df.to_excel(writer, sheet_name="Element Level Errors", index=False)
            
            # Overall statistics
            if self.error_metrics["overall"]:
                overall_data = []
                for corr_type, stats in self.error_metrics["overall"].items():
                    overall_data.append({
                        "Corrosion Type": corr_type,
                        "Mean Abs Error (%)": stats["mean_abs_error_%"],
                        "Std Abs Error (%)": stats["std_abs_error_%"],
                        "Max Abs Error (%)": stats["max_abs_error_%"],
                        "Mean Rel Error (%)": stats["mean_rel_error_%"],
                        "Num Comparisons": stats["num_comparisons"]
                    })
                pd.DataFrame(overall_data).to_excel(writer, sheet_name="Overall Statistics", index=False)
            
            # Matching statistics
            matching_data = [{
                "Metric": "Total GT Instances",
                "Value": self.matching_stats["total_gt_instances"]
            }, {
                "Metric": "Total Pred Instances",
                "Value": self.matching_stats["total_pred_instances"]
            }, {
                "Metric": "Matched Instances",
                "Value": self.matching_stats["matched_instances"]
            }, {
                "Metric": "Unmatched GT (False Neg)",
                "Value": self.matching_stats["unmatched_gt"]
            }, {
                "Metric": "Unmatched Pred (False Pos)",
                "Value": self.matching_stats["unmatched_pred"]
            }, {
                "Metric": "Recall (%)",
                "Value": self.matching_stats["matched_instances"] / max(1, self.matching_stats["total_gt_instances"]) * 100
            }, {
                "Metric": "Precision (%)",
                "Value": self.matching_stats["matched_instances"] / max(1, self.matching_stats["total_pred_instances"]) * 100
            }]
            pd.DataFrame(matching_data).to_excel(writer, sheet_name="Matching Statistics", index=False)
            
            # False negatives
            if self.matching_stats["false_negatives"]:
                df_fn = pd.DataFrame(self.matching_stats["false_negatives"])
                df_fn.to_excel(writer, sheet_name="False Negatives", index=False)
            
            # False positives
            if self.matching_stats["false_positives"]:
                df_fp = pd.DataFrame(self.matching_stats["false_positives"])
                df_fp.to_excel(writer, sheet_name="False Positives", index=False)
        
        logger.info(f"Saved Excel report to {excel_path}")
    
    def _generate_plots(self):
        """Generate error analysis plots."""
        if not self.error_metrics["overall"]:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Error Analysis Report ({self.dataset_name})", fontsize=16, fontweight='bold')
        
        corr_types = ["Fair", "Poor", "Severe"]
        colors = ['green', 'yellow', 'red']
        
        # Mean absolute errors
        mean_errors = [self.error_metrics["overall"].get(ct, {}).get("mean_abs_error_%", 0) for ct in corr_types]
        axes[0, 0].bar(corr_types, mean_errors, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Mean Absolute Error (%)')
        axes[0, 0].set_title('Mean Absolute Error by Corrosion Type\n(IoU-Matched Instances)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distributions
        error_dists = []
        for ct in corr_types:
            if ct in self.error_metrics["per_corrosion_type"]:
                errors = [e["abs_error"] * 100 for e in self.error_metrics["per_corrosion_type"][ct]]
                error_dists.append(errors)
            else:
                error_dists.append([])
        
        bp = axes[0, 1].boxplot(error_dists, tick_labels=corr_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 1].set_ylabel('Absolute Error (%)')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GT vs Pred scatter
        for ct, color in zip(corr_types, colors):
            if ct in self.error_metrics["per_corrosion_type"]:
                gt_vals = [e["gt"] * 100 for e in self.error_metrics["per_corrosion_type"][ct]]
                pred_vals = [e["pred"] * 100 for e in self.error_metrics["per_corrosion_type"][ct]]
                axes[1, 0].scatter(gt_vals, pred_vals, alpha=0.6, label=ct, color=color, s=50)
        
        axes[1, 0].plot([0, 100], [0, 100], 'k--', label='Perfect', linewidth=2)
        axes[1, 0].set_xlabel('Ground Truth (%)')
        axes[1, 0].set_ylabel('Predicted (%)')
        axes[1, 0].set_title('GT vs Predicted Ratios')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Matching statistics
        matching_labels = ['GT\nInstances', 'Pred\nInstances', 'Matched', 'Unmatched\nGT', 'Unmatched\nPred']
        matching_values = [
            self.matching_stats["total_gt_instances"],
            self.matching_stats["total_pred_instances"],
            self.matching_stats["matched_instances"],
            self.matching_stats["unmatched_gt"],
            self.matching_stats["unmatched_pred"]
        ]
        matching_colors = ['blue', 'cyan', 'green', 'red', 'orange']
        
        axes[1, 1].bar(matching_labels, matching_values, color=matching_colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Instance Matching Statistics (IoU ≥ {self.iou_threshold})')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.error_dir, "error_analysis_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plots to {plot_path}")
    
    def _generate_report(self):
        """Generate markdown report."""
        report_path = os.path.join(self.reports_dir, "error_analysis_summary.md")

        with open(report_path, 'w') as f:
            f.write(f"# Error Analysis Report ({self.dataset_name})\n")
            f.write("## (IoU-Based Instance Matching)\n\n")
            
            f.write("This analysis uses **IoU-based matching** to ensure GT and predicted instances ")
            f.write(f"are correctly paired (IoU ≥ {self.iou_threshold}).\n\n")
            
            f.write("## Instance Matching Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total GT Instances | {self.matching_stats['total_gt_instances']} |\n")
            f.write(f"| Total Predicted Instances | {self.matching_stats['total_pred_instances']} |\n")
            f.write(f"| Matched Instances | {self.matching_stats['matched_instances']} |\n")
            f.write(f"| Unmatched GT (False Negatives) | {self.matching_stats['unmatched_gt']} |\n")
            f.write(f"| Unmatched Pred (False Positives) | {self.matching_stats['unmatched_pred']} |\n")
            
            recall = self.matching_stats["matched_instances"] / max(1, self.matching_stats["total_gt_instances"]) * 100
            precision = self.matching_stats["matched_instances"] / max(1, self.matching_stats["total_pred_instances"]) * 100
            f.write(f"| **Recall** | **{recall:.2f}%** |\n")
            f.write(f"| **Precision** | **{precision:.2f}%** |\n\n")
            
            f.write("## Corrosion Prediction Errors (Matched Instances Only)\n\n")
            f.write("| Corrosion Type | Mean Abs Error (%) | Std | Max | Comparisons |\n")
            f.write("|----------------|--------------------|--------------------|-------------------|--------------|\n")
            
            for corr_type in ["Fair", "Poor", "Severe"]:
                if corr_type in self.error_metrics["overall"]:
                    stats = self.error_metrics["overall"][corr_type]
                    f.write(f"| {corr_type} | {stats['mean_abs_error_%']:.2f} | ")
                    f.write(f"{stats['std_abs_error_%']:.2f} | {stats['max_abs_error_%']:.2f} | ")
                    f.write(f"{stats['num_comparisons']} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            if self.error_metrics["overall"]:
                best = min(self.error_metrics["overall"].items(), key=lambda x: x[1]["mean_abs_error_%"])
                worst = max(self.error_metrics["overall"].items(), key=lambda x: x[1]["mean_abs_error_%"])
                
                f.write(f"- **Best Performance**: {best[0]} (Mean Error: {best[1]['mean_abs_error_%']:.2f}%)\n")
                f.write(f"- **Worst Performance**: {worst[0]} (Mean Error: {worst[1]['mean_abs_error_%']:.2f}%)\n")
                f.write(f"- **Instance Detection Recall**: {recall:.2f}% (model found this % of GT elements)\n")
                f.write(f"- **Instance Detection Precision**: {precision:.2f}% (this % of predictions matched GT)\n\n")
            
            f.write("## Interpretation\n\n")
            f.write("These metrics are calculated **only for correctly matched instances** (IoU ≥ 0.5), ")
            f.write("providing accurate assessment of corrosion prediction quality.\n\n")
            f.write("Unmatched instances represent:\n")
            f.write("- **False Negatives**: GT elements the model failed to detect\n")
            f.write("- **False Positives**: Model predictions with no corresponding GT\n\n")
            
            f.write("## Output Files\n\n")
            f.write("- `matching_statistics.json`: Instance matching details\n")
            f.write("- `element_level_errors.csv`: Per-element comparison\n")
            f.write("- `error_analysis.xlsx`: Comprehensive Excel report with multiple sheets\n")
            f.write("- `error_analysis_plots.png`: Visualization of errors and matching\n")
            f.write("- `visualizations/`: Per-image comparison visualizations (2 types per image):\n")
            f.write("  - `overview_*.png`: 2x2 grid showing semantic + instance results\n")
            f.write("  - `elementwise_*.png`: Side-by-side GT vs Pred element-wise corrosion (KEY VISUALIZATION!)\n")
        
        logger.info(f"Saved report to {report_path}")


def setup_semantic_cfg(args):
    """Setup semantic segmentation config."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.semantic_config)
    if args.semantic_weights:
        cfg.MODEL.WEIGHTS = args.semantic_weights
    return cfg


def setup_instance_cfg(args):
    """Setup instance segmentation config."""
    cfg = get_cfg()
    add_deeplab_config(cfg)

    cfg.SOLVER.BETAS = [0.9, 0.999]
    
    # Register TEST keys
    if not hasattr(cfg, "TEST"):
        from fvcore.common.config import CfgNode as CN
        cfg.TEST = CN()
    
    if not hasattr(cfg.TEST, "SEMANTIC_ON"):
        cfg.TEST.SEMANTIC_ON = False
    if not hasattr(cfg.TEST, "INSTANCE_ON"):
        cfg.TEST.INSTANCE_ON = True
    if not hasattr(cfg.TEST, "PANOPTIC_ON"):
        cfg.TEST.PANOPTIC_ON = False
    
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.instance_config)
    if args.instance_weights:
        cfg.MODEL.WEIGHTS = args.instance_weights
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Error Analysis with IoU-Based Matching"
    )
    parser.add_argument("--semantic-config", required=True, help="Semantic config file")
    parser.add_argument("--semantic-weights", required=True, help="Semantic model weights")
    parser.add_argument("--instance-config", required=True, help="Instance config file")
    parser.add_argument("--instance-weights", required=True, help="Instance model weights")
    
    # Remove arguments that will be dynamically set
    # parser.add_argument("--instance-gt-json", required=True, help="Instance GT COCO JSON")
    # parser.add_argument("--semantic-gt-dir", required=True, help="Semantic GT mask directory")
    # parser.add_argument("--images-dir", required=True, help="Images directory")
    # parser.add_argument("--dataset-name", default="corrosion_val", help="Dataset name")
    
    parser.add_argument("--output-dir", default="./output/error_analysis", help="Base output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence for predictions")
    parser.add_argument("--output-format", choices=["json", "csv", "excel", "all"], default="all")
    parser.add_argument("--generate-visualizations", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Register datasets
    datasets_semantic.register_semantic_datasets()
    datasets.register_datasets()
    
    # Setup base configs
    semantic_cfg = setup_semantic_cfg(args)
    instance_cfg = setup_instance_cfg(args)
    
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS WITH IOU-BASED MATCHING")
    logger.info("=" * 60)
    
    # Define splits to analyze
    splits_to_analyze = ["val", "test"]
    logger.info(f"Scheduled to analyze splits: {splits_to_analyze}")
    
    for split in splits_to_analyze:
        logger.info("="*80)
        logger.info(f"🚀 STARTING ERROR ANALYSIS FOR SPLIT: {split}")
        logger.info("="*80)
        
        # Dynamically construct paths
        dataset_name = f"corrosion_{split}"
        instance_dataset_name = f"instance_{split}"
        
        instance_gt_json = os.path.join(PROJECT_ROOT, "datasets", "instance", split, "annotations.json")
        semantic_gt_dir = os.path.join(PROJECT_ROOT, "datasets", "semantic", split, "annotations", "sem_seg")
        images_dir = os.path.join(PROJECT_ROOT, "datasets", "instance", split, "images")
        
        split_output_dir = os.path.join(base_output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Check if paths exist
        if not os.path.exists(instance_gt_json):
            logger.error(f"Missing {split} instance GT: {instance_gt_json}. Skipping split.")
            continue
        if not os.path.exists(semantic_gt_dir):
            logger.error(f"Missing {split} semantic GT dir: {semantic_gt_dir}. Skipping split.")
            continue
        if not os.path.exists(images_dir):
            logger.error(f"Missing {split} images dir: {images_dir}. Skipping split.")
            continue
        
        # Run analysis for this split
        analyzer = ErrorAnalyzer(
            semantic_cfg=semantic_cfg.clone(),
            instance_cfg=instance_cfg.clone(),
            instance_gt_json=instance_gt_json,
            semantic_gt_dir=semantic_gt_dir,
            images_dir=images_dir,
            output_dir=split_output_dir,
            dataset_name=dataset_name,
            instance_dataset_name=instance_dataset_name,
            iou_threshold=args.iou_threshold,
            min_confidence=args.min_confidence
        )
        
        analyzer.analyze_all_images(generate_visualizations=args.generate_visualizations)
        analyzer.save_results(output_format=args.output_format)
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE FOR ALL SPLITS")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {base_output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
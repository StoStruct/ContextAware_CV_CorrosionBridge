#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import time
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix
from PIL import Image  # Added for High PPI saving

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.evaluation import SemSegEvaluator, DatasetEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.detection_utils import read_image
from detectron2.data import transforms as T

# Add Mask2Former path
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
sys.path.insert(0, MASK2FORMER_ROOT)

from mask2former import add_maskformer2_config, SemanticSegmentorWithTTA

# Import dataset registration
import datasets_semantic

# Setup logger
setup_logger()
logger = logging.getLogger("detectron2.semantic_evaluation")

# Custom mapper for semantic segmentation testing
class SemanticSegTestingMapper(DatasetMapper):
    """
    A mapper that handles semantic segmentation testing.
    """
    def __init__(self, cfg, is_train=False):
        super().__init__(cfg, is_train=is_train)
        
        # Store image format
        self.img_format = cfg.INPUT.FORMAT
        
        # Semantic segmentation specific changes
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = dataset_dict.copy()
        
        # Read image
        image = read_image(dataset_dict["file_name"], format=self.img_format)
        
        # No augmentations - just use the original image
        # Instead of using self._apply_augmentations
        # Handle augmentations carefully
        if self.augmentations:
            # Check if it's an AugmentationList and extract augs if needed
            augs = []
            if hasattr(self.augmentations, 'augs'):
                augs = self.augmentations.augs
            else:
                augs = self.augmentations
                
            image, transforms = T.apply_transform_gens(augs, image)
        else:
            transforms = None
        
        # Transform image
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        # Apply the same transformations to the semantic segmentation map
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = read_image(dataset_dict["sem_seg_file_name"], format="L")
            if transforms is not None:
                sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        
        # Remove unnecessary fields
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
        
        return dataset_dict

class SemanticSegmentationEvaluator:
    """Comprehensive evaluator for semantic segmentation models."""
    
    def __init__(self, cfg, dataset_name: str, output_dir: str, model_name: str = "mask2former_semantic"):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory structure
        self.vis_dir = os.path.join(output_dir, "visualizations")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load metadata and dataset
        self.metadata = MetadataCatalog.get(dataset_name)
        self.dataset = DatasetCatalog.get(dataset_name)
        
        # Build model and load weights
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        # Class names and mapping
        if hasattr(self.metadata, "stuff_classes"):
            self.class_names = self.metadata.stuff_classes
        else:
            self.class_names = [f"class_{i}" for i in range(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)]
        
        # Initialize evaluator
        self.evaluator = SemSegEvaluator(
            dataset_name,
            distributed=False,
            output_dir=self.metrics_dir,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        
        # Store evaluation results
        self.results = {}
        self.predictions = []
        self.ground_truth = []
        
        logger.info(f"Initialized SemanticSegmentationEvaluator for dataset: {dataset_name}")
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
    
    def evaluate(self) -> Dict:
        """Run full evaluation process."""
        # Create test loader with appropriate mapper
        mapper = SemanticSegTestingMapper(self.cfg, is_train=False)
        loader = build_detection_test_loader(self.cfg, self.dataset_name, mapper=mapper)
        
        # Evaluate using standard evaluator
        results = inference_on_dataset(self.model, loader, self.evaluator)
        self.results["standard"] = results
        
        # Process and match predictions with ground truth
        self._process_predictions_and_ground_truth()
        
        # Calculate confusion matrix
        self._calculate_confusion_matrix()
        
        # Calculate F1 scores
        self._calculate_f1_scores()
        
        # Calculate per-image IoU
        self._calculate_per_image_iou()
        
        # Calculate boundary accuracy
        self.analyze_boundary_accuracy()
        
        # Analyze size sensitivity
        self.analyze_size_sensitivity()
        
        # Analyze model complexity
        self.analyze_model_complexity()
        
        # Visualize training history
        self.visualize_training_history()
        
        # Visualize worst cases
        self.visualize_worst_cases(num_samples=10)
        
        # Generate visualizations
        self._visualize_predictions()
        
        # Generate plots
        self._plot_confusion_matrix()
        self._plot_f1_scores()
        self._plot_per_image_iou()
        
        # Save overall results
        self._save_results()
        
        return self.results

    def _process_predictions_and_ground_truth(self):
        """Process all images, make predictions and organize ground truth."""
        logger.info("Processing predictions and ground truth...")
        
        # Create test loader with appropriate mapper
        mapper = SemanticSegTestingMapper(self.cfg, is_train=False)
        loader = build_detection_test_loader(self.cfg, self.dataset_name, mapper=mapper)
        
        # Process all images
        for batch in tqdm(loader):
            for data in batch:
                # Get image and file name
                file_name = data["file_name"]
                image_id = data.get("image_id", os.path.basename(file_name))
                
                # Read image
                image = read_image(file_name, format=self.cfg.INPUT.FORMAT)
                
                # Get ground truth semantic segmentation
                if "sem_seg" in data:
                    gt_sem_seg = data["sem_seg"].numpy()
                else:
                    continue  # Skip if no ground truth
                
                # Make prediction
                with torch.no_grad():
                    # Prepare input
                    height, width = image.shape[:2]
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    inputs = {"image": image, "height": height, "width": width}
                    
                    # Run model
                    outputs = self.model([inputs])[0]
                    
                    # Get semantic segmentation prediction
                    if "sem_seg" in outputs:
                        pred_sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
                    else:
                        continue  # Skip if no prediction
                
                # Store ground truth and prediction
                self.ground_truth.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "sem_seg": gt_sem_seg
                })
                
                self.predictions.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "sem_seg": pred_sem_seg
                })
        
        logger.info(f"Processed {len(self.predictions)} images")

    def _calculate_confusion_matrix(self):
        """Calculate confusion matrix for semantic segmentation."""
        logger.info("Calculating confusion matrix...")
        
        if not self.ground_truth or not self.predictions:
            logger.warning("No data available to calculate confusion matrix")
            return
        
        # Initialize confusion matrix
        num_classes = len(self.class_names)
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # Process each image
        for gt, pred in zip(self.ground_truth, self.predictions):
            gt_seg = gt["sem_seg"]
            pred_seg = pred["sem_seg"]
            
            # Ensure same shape
            if gt_seg.shape != pred_seg.shape:
                logger.info(f"Resizing ground truth for {gt['file_name']}: {gt_seg.shape} to {pred_seg.shape}")
                # Resize ground truth to match prediction dimensions
                pred_h, pred_w = pred_seg.shape
                gt_seg = cv2.resize(gt_seg, (pred_w, pred_h), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Calculate confusion matrix for this image
            mask = (gt_seg >= 0) & (gt_seg < num_classes)
            
            # Flatten arrays for confusion matrix calculation
            gt_seg_flat = gt_seg[mask].flatten()
            pred_seg_flat = pred_seg[mask].flatten()
            
            # Update confusion matrix
            image_conf_matrix = confusion_matrix(
                gt_seg_flat, pred_seg_flat, 
                labels=list(range(num_classes))
            )
            conf_matrix += image_conf_matrix
        
        # Create normalized confusion matrix (by row)
        row_sums = conf_matrix.sum(axis=1)
        conf_matrix_normalized = np.zeros_like(conf_matrix, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                conf_matrix_normalized[i] = conf_matrix[i] / row_sums[i]
        
        # Store results
        self.results["confusion_matrix"] = {
            "matrix": conf_matrix,
            "matrix_normalized": conf_matrix_normalized,
            "class_names": self.class_names
        }
        
        # Print confusion matrix (absolute counts)
        logger.info(f"Confusion Matrix (counts):")
        headers = [""] + self.class_names
        table = []
        for i, class_name in enumerate(self.class_names):
            row = [class_name] + [str(conf_matrix[i, j]) for j in range(num_classes)]
            table.append(row)
        
        logger.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))
        
        # Print normalized confusion matrix (percentages)
        logger.info(f"Confusion Matrix (normalized by row, %):")
        table = []
        for i, class_name in enumerate(self.class_names):
            row = [class_name] + [f"{conf_matrix_normalized[i, j]*100:.1f}%" for j in range(num_classes)]
            table.append(row)
        
        logger.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))

    def _calculate_f1_scores(self):
        """Calculate precision, recall, and F1 score for each class."""
        logger.info("Calculating F1 scores...")
        
        if "confusion_matrix" not in self.results:
            logger.warning("No confusion matrix available to calculate F1 scores")
            return
        
        conf_matrix = self.results["confusion_matrix"]["matrix"]
        
        # Initialize metrics
        precision = {}
        recall = {}
        f1_score = {}
        
        # Calculate for each class
        for i, class_name in enumerate(self.class_names):
            # True positives: diagonal element
            tp = conf_matrix[i, i]
            
            # False positives: sum of column i (excluding true positives)
            fp = conf_matrix[:, i].sum() - tp
            
            # False negatives: sum of row i (excluding true positives)
            fn = conf_matrix[i, :].sum() - tp
            
            # Calculate precision, recall, F1
            if tp + fp > 0:
                precision[class_name] = tp / (tp + fp)
            else:
                precision[class_name] = 0
                
            if tp + fn > 0:
                recall[class_name] = tp / (tp + fn)
            else:
                recall[class_name] = 0
                
            if precision[class_name] + recall[class_name] > 0:
                f1_score[class_name] = 2 * precision[class_name] * recall[class_name] / (precision[class_name] + recall[class_name])
            else:
                f1_score[class_name] = 0
        
        # Calculate macro average
        macro_precision = sum(precision.values()) / len(precision)
        macro_recall = sum(recall.values()) / len(recall)
        macro_f1 = sum(f1_score.values()) / len(f1_score)
        
        # Store results
        self.results["f1_scores"] = {
            "per_class": f1_score,
            "macro_avg": macro_f1
        }
        
        self.results["precision"] = {
            "per_class": precision,
            "macro_avg": macro_precision
        }
        
        self.results["recall"] = {
            "per_class": recall,
            "macro_avg": macro_recall
        }
        
        # Print results
        logger.info(f"F1 Scores:")
        for class_name, f1 in f1_score.items():
            logger.info(f" {class_name}: {f1:.4f} (P: {precision[class_name]:.4f}, R: {recall[class_name]:.4f})")
        
        logger.info(f"Macro F1: {macro_f1:.4f} (P: {macro_precision:.4f}, R: {macro_recall:.4f})")
    
    def _calculate_per_image_iou(self):
        """Calculate IoU for each image and class."""
        logger.info("Calculating per-image IoU...")
        
        if not self.ground_truth or not self.predictions:
            logger.warning("No data available to calculate per-image IoU")
            return
        
        # Initialize storage for per-image IoU
        image_ious = []
        class_ious = defaultdict(list)
        
        # Process each image
        for gt, pred in zip(self.ground_truth, self.predictions):
            gt_seg = gt["sem_seg"]
            pred_seg = pred["sem_seg"]
            
            # Ensure same shape
            if gt_seg.shape != pred_seg.shape:
                logger.info(f"Resizing ground truth for {gt['file_name']}: {gt_seg.shape} to {pred_seg.shape}")
                # Resize ground truth to match prediction dimensions
                pred_h, pred_w = pred_seg.shape
                gt_seg = cv2.resize(gt_seg, (pred_w, pred_h), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Calculate IoU for each class in this image
            image_iou = {}
            for i, class_name in enumerate(self.class_names):
                # Create binary masks for this class
                gt_mask = (gt_seg == i)
                pred_mask = (pred_seg == i)
                
                # Calculate intersection and union
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                
                # Calculate IoU (handle division by zero)
                if union > 0:
                    iou = intersection / union
                else:
                    # If both ground truth and prediction don't have this class,
                    # we consider IoU as NaN (not applicable)
                    iou = np.nan
                
                image_iou[class_name] = float(iou)
                
                # Store for class-wise statistics (ignore NaN values)
                if not np.isnan(iou):
                    class_ious[class_name].append(iou)
            
            # Store per-image IoU
            image_ious.append({
                "image_id": gt["image_id"],
                "file_name": gt["file_name"],
                "iou": image_iou
            })
        
        # Calculate mean IoU for each class
        class_mean_iou = {}
        for class_name, ious in class_ious.items():
            if ious:
                class_mean_iou[class_name] = float(np.mean(ious))
            else:
                class_mean_iou[class_name] = 0.0
        
        # Calculate mean IoU across all images
        mean_iou = np.mean([
            np.nanmean(list(img["iou"].values()))
            for img in image_ious
        ])
        
        # Store results
        self.results["per_image_iou"] = {
            "images": image_ious,
            "class_mean": class_mean_iou,
            "mean": float(mean_iou)
        }
        
        # Print results
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        logger.info(f"Class-wise Mean IoU:")
        for class_name, iou in class_mean_iou.items():
            logger.info(f" {class_name}: {iou:.4f}")
    
    def analyze_boundary_accuracy(self):
        """Analyze segmentation boundary accuracy using trimap analysis."""
        logger.info("Analyzing boundary accuracy...")
        
        if not self.ground_truth or not self.predictions:
            logger.warning("No data available for boundary analysis")
            return
        
        # Define boundary widths (in pixels)
        boundary_widths = [1, 2, 3, 5, 10]
        class_boundary_ious = {width: {cls: [] for cls in self.class_names} for width in boundary_widths}
        
        # Process each image
        for gt, pred in zip(self.ground_truth, self.predictions):
            gt_seg = gt["sem_seg"]
            pred_seg = pred["sem_seg"]
            
            # Ensure same shape
            if gt_seg.shape != pred_seg.shape:
                pred_h, pred_w = pred_seg.shape
                gt_seg = cv2.resize(gt_seg, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
            
            # Process each class
            for class_idx, class_name in enumerate(self.class_names):
                # Create binary masks for this class
                gt_mask = (gt_seg == class_idx).astype(np.uint8)
                pred_mask = (pred_seg == class_idx).astype(np.uint8)
                
                # Skip if class not present in ground truth
                if not np.any(gt_mask):
                    continue
                
                # Find boundaries using morphological operations
                for width in boundary_widths:
                    # Create kernel for dilation
                    kernel = np.ones((width, width), np.uint8)
                    
                    # Dilate ground truth mask
                    gt_dilated = cv2.dilate(gt_mask, kernel, iterations=1)
                    
                    # Create trimap by subtracting original from dilated
                    boundary_region = gt_dilated - gt_mask
                    
                    # Skip if no boundary region
                    if not np.any(boundary_region):
                        continue
                    
                    # Calculate IoU in boundary region only
                    intersection = np.logical_and(gt_mask == pred_mask, boundary_region).sum()
                    union = np.logical_or(gt_mask != pred_mask, boundary_region).sum()
                    
                    if union > 0:
                        boundary_iou = intersection / float(union)
                        class_boundary_ious[width][class_name].append(boundary_iou)
        
        # Calculate mean boundary IoU for each width and class
        boundary_results = {}
        for width in boundary_widths:
            class_results = {}
            valid_iou_classes = 0
            total_iou = 0.0
            
            for class_name in self.class_names:
                if class_boundary_ious[width][class_name]:
                    class_results[class_name] = np.mean(class_boundary_ious[width][class_name])
                    valid_iou_classes += 1
                    total_iou += class_results[class_name]
                else:
                    class_results[class_name] = 0.0
            
            # Calculate mean across all classes (handling case when no valid classes)
            if valid_iou_classes > 0:
                mean_iou = total_iou / valid_iou_classes
            else:
                mean_iou = float('nan')  # Use NaN when no valid data
                
            boundary_results[width] = {
                "per_class": class_results,
                "mean": float(mean_iou)
            }
        
        # Store results
        self.results["boundary_analysis"] = boundary_results
        
        # Log results
        logger.info("Boundary IoU Results:")
        for width, results in boundary_results.items():
            logger.info(f"Boundary width {width}px: Mean IoU = {results['mean']:.4f}" if not np.isnan(results['mean']) 
                    else f"Boundary width {width}px: Mean IoU = nan (no valid data)")
        
        # Plot boundary IoU by width
        self._plot_boundary_analysis()
        
        return boundary_results
    
    def analyze_size_sensitivity(self):
        """Analyze model performance on different image sizes."""
        logger.info("Analyzing size sensitivity...")
        
        # Skip if no data available
        if not self.ground_truth or not self.predictions:
            logger.warning("No data available for size sensitivity analysis")
            return
        
        # Group images by size - FIX: Added "iou" key to the large bucket
        size_buckets = {
            "small": {"width": (0, 512), "height": (0, 512), "images": [], "iou": []},
            "medium": {"width": (513, 1024), "height": (513, 1024), "images": [], "iou": []},
            "large": {"width": (1025, float('inf')), "height": (1025, float('inf')), "images": [], "iou": []}
        }
        
        # Process each image
        for i, (gt, pred) in enumerate(zip(self.ground_truth, self.predictions)):
            file_name = gt["file_name"]
            img = cv2.imread(file_name)
            if img is None:
                continue
                
            height, width = img.shape[:2]
            
            # Get mean IoU for this image
            if "per_image_iou" in self.results and i < len(self.results["per_image_iou"]["images"]):
                img_iou = np.nanmean(list(self.results["per_image_iou"]["images"][i]["iou"].values()))
            else:
                # Recalculate IoU if not available
                gt_seg = gt["sem_seg"]
                pred_seg = pred["sem_seg"]
                
                # Ensure same shape
                if gt_seg.shape != pred_seg.shape:
                    pred_h, pred_w = pred_seg.shape
                    gt_seg = cv2.resize(gt_seg, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate mean IoU across all classes
                ious = []
                for class_idx in range(len(self.class_names)):
                    gt_mask = (gt_seg == class_idx)
                    pred_mask = (pred_seg == class_idx)
                    
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()
                    
                    if union > 0:
                        ious.append(intersection / union)
                
                img_iou = np.mean(ious) if ious else 0.0
            
            # Assign to bucket based on max dimension
            max_dim = max(width, height)
            if max_dim <= 512:
                size_buckets["small"]["images"].append(file_name)
                size_buckets["small"]["iou"].append(img_iou)
            elif max_dim <= 1024:
                size_buckets["medium"]["images"].append(file_name)
                size_buckets["medium"]["iou"].append(img_iou)
            else:
                size_buckets["large"]["images"].append(file_name)
                size_buckets["large"]["iou"].append(img_iou)
        
        # Calculate statistics for each bucket
        size_sensitivity = {}
        for size, bucket in size_buckets.items():
            if bucket["iou"]:
                size_sensitivity[size] = {
                    "count": len(bucket["images"]),
                    "mean_iou": float(np.mean(bucket["iou"])),
                    "median_iou": float(np.median(bucket["iou"])),
                    "std_iou": float(np.std(bucket["iou"])) if len(bucket["iou"]) > 1 else 0.0
                }
            else:
                size_sensitivity[size] = {
                    "count": 0,
                    "mean_iou": 0.0,
                    "median_iou": 0.0,
                    "std_iou": 0.0
                }
        
        # Store results
        self.results["size_sensitivity"] = size_sensitivity
        
        # Log results
        logger.info("Size Sensitivity Results:")
        for size, stats in size_sensitivity.items():
            logger.info(f"{size.capitalize()} images ({stats['count']}): Mean IoU = {stats['mean_iou']:.4f}, " + 
                    f"Median IoU = {stats['median_iou']:.4f}, Std = {stats['std_iou']:.4f}")
        
        # Plot results
        self._plot_size_sensitivity()
        
        return size_sensitivity
    
    def _plot_boundary_analysis(self):
        """Plot boundary IoU analysis results."""
        if "boundary_analysis" not in self.results:
            return
        
        boundary_results = self.results["boundary_analysis"]
        
        # Plot mean boundary IoU by width
        plt.figure(figsize=(10, 6))
        widths = sorted(boundary_results.keys())
        mean_ious = []
        valid_widths = []
        
        # Filter out NaN values
        for w in widths:
            if not np.isnan(boundary_results[w]["mean"]):
                mean_ious.append(boundary_results[w]["mean"])
                valid_widths.append(w)
        
        # Skip plotting if no valid data
        if not valid_widths:
            logger.warning("No valid boundary IoU data to plot")
            return
            
        plt.plot(valid_widths, mean_ious, 'o-', linewidth=2)
        plt.xlabel('Boundary Width (pixels)')
        plt.ylabel('Mean Boundary IoU')
        plt.title('Boundary Accuracy Analysis')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(valid_widths)
        
        # Add value labels (only for valid values)
        for x, y in zip(valid_widths, mean_ious):
            # Check if value is finite before adding text
            if np.isfinite(y):
                plt.text(x, y + 0.01, f'{y:.3f}', ha='center')
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'boundary_analysis.png'), dpi=300)
        plt.close()
        
        # Plot per-class boundary IoU for a specific width (e.g., 3px)
        width = 3  # Choose a reasonable width that typically has data
        if width in boundary_results:
            plt.figure(figsize=(12, 6))
            class_ious = boundary_results[width]["per_class"]
            
            # Sort classes by IoU for better visualization
            sorted_classes = sorted(class_ious.items(), key=lambda x: x[1], reverse=True)
            class_names = [c[0] for c in sorted_classes if c[1] > 0 and np.isfinite(c[1])]
            iou_values = [c[1] for c in sorted_classes if c[1] > 0 and np.isfinite(c[1])]
            
            if not class_names:  # Skip if no valid classes
                return
                
            bars = plt.bar(class_names, iou_values)
            plt.xlabel('Class')
            plt.ylabel(f'Boundary IoU (width={width}px)')
            plt.title(f'Per-Class Boundary IoU (width={width}px)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels (only for valid values)
            for bar in bars:
                height = bar.get_height()
                if np.isfinite(height):  # Check if height is finite
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'boundary_analysis_per_class_width{width}.png'), dpi=300)
            plt.close()

    def _plot_size_sensitivity(self):
        """Plot size sensitivity analysis results."""
        if "size_sensitivity" not in self.results:
            return
        
        size_sensitivity = self.results["size_sensitivity"]
        
        # Extract data
        sizes = list(size_sensitivity.keys())
        counts = [size_sensitivity[s]["count"] for s in sizes]
        mean_ious = [size_sensitivity[s]["mean_iou"] for s in sizes]
        
        # Filter out sizes with no images
        valid_indices = [i for i, count in enumerate(counts) if count > 0]
        if not valid_indices:
            logger.warning("No valid size categories with images to plot")
            return
            
        sizes = [sizes[i] for i in valid_indices]
        counts = [counts[i] for i in valid_indices]
        mean_ious = [mean_ious[i] for i in valid_indices]
        
        # Create plot with single figure creation
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Calculate bar positions
        bar_width = 0.35
        index = np.arange(len(sizes))
        
        # Plot mean IoU
        bars1 = ax1.bar(index, mean_ious, bar_width, label='Mean IoU', color='tab:blue')
        ax1.set_ylabel('Mean IoU', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim([0, 1.0])
        
        # Create second y-axis for counts
        ax2 = ax1.twinx()
        bars2 = ax2.bar(index + bar_width, counts, bar_width, label='Image Count', color='tab:orange', alpha=0.7)
        ax2.set_ylabel('Image Count', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # Add labels and title
        ax1.set_xlabel('Image Size')
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels([s.capitalize() for s in sizes])
        plt.title('Model Performance by Image Size')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if np.isfinite(height):  # Ensure height is a valid number
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='tab:blue')
        
        for bar in bars2:
            height = bar.get_height()
            if np.isfinite(height):  # Ensure height is a valid number
                ax2.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='tab:orange')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'size_sensitivity.png'), dpi=300)
        plt.close(fig)  # Explicitly close this figure

    def _custom_draw_sem_seg(self, img, sem_seg, alpha=0.7, background_alpha=0.3):
        """
        Custom function to draw semantic segmentation with transparent background
        and no class names on the image.
        
        Args:
            img: Input image (H, W, 3) in RGB
            sem_seg: Semantic segmentation (H, W) with class indices
            alpha: Overall alpha/opacity for non-background classes
            background_alpha: Alpha/opacity specifically for background class
            
        Returns:
            Visualization image with segmentation overlay
        """
        img = img.copy()
        h, w = sem_seg.shape
        
        # Create empty colored mask
        colored_mask = np.zeros((h, w, 4), dtype=np.float32)
        
        # FORCE colors as requested:
        # 0: Background - Very very light green (225, 255, 225)
        # 1: Fair - Yellow (255, 255, 0)
        # 2: Poor - Orange (255, 165, 0)
        # 3: Severe - Red (255, 0, 0)
        colors = [
            (225, 255, 225), # Background - Very very light green
            (255, 255, 0),   # Fair - Yellow
            (255, 165, 0),   # Poor - Orange
            (255, 0, 0)      # Severe - Red
        ]
        
        # Fill colored mask for each class
        for class_id, color in enumerate(colors):
            # Find pixels of this class
            mask = (sem_seg == class_id)
            if not np.any(mask):
                continue
                
            # Use different alpha for background vs other classes
            current_alpha = background_alpha if class_id == 0 else alpha
            
            # Set color with appropriate alpha
            colored_mask[mask] = (*color, 255 * current_alpha)
        
        # Convert RGBA mask to RGB with transparency
        # First create a blank RGB image
        result = img.copy()
        
        # For each pixel with non-zero alpha, blend with original image
        mask_rgb = colored_mask[..., :3]
        mask_alpha = colored_mask[..., 3:4] / 255.0
        
        # Apply alpha blending
        np.multiply(result, 1.0 - mask_alpha, out=result, casting="unsafe")
        np.add(result, mask_rgb * mask_alpha, out=result, casting="unsafe")
        
        return result.astype(np.uint8)
    
    def _visualize_predictions(self):
        """Generate visualizations for each image."""
        logger.info("Generating visualizations...")
        
        # Limit the number of visualizations to avoid generating too many files
        max_vis = min(50, len(self.ground_truth))
        
        # Process each image
        for i in range(max_vis):
            gt = self.ground_truth[i]
            pred = self.predictions[i]
            
            # Read image
            file_name = gt["file_name"]
            img = cv2.imread(file_name)
            if img is None:
                logger.warning(f"Failed to read image: {file_name}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create visualizers with modified setup
            vis_gt = Visualizer(img.copy(), metadata=self.metadata, scale=1.0)
            vis_pred = Visualizer(img.copy(), metadata=self.metadata, scale=1.0)
            
            # Resize ground truth mask to match prediction dimensions if needed
            gt_sem_seg = gt["sem_seg"]
            pred_sem_seg = pred["sem_seg"]
            if gt_sem_seg.shape != pred_sem_seg.shape:
                # Resize ground truth to match prediction
                pred_h, pred_w = pred_sem_seg.shape
                gt_sem_seg = cv2.resize(gt_sem_seg, (pred_w, pred_h), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Create custom visualization with semi-transparent background
            # 1. Create custom visualization for ground truth
            gt_viz = self._custom_draw_sem_seg(
                img.copy(), 
                gt_sem_seg,
                alpha=0.7,  # Overall alpha
                background_alpha=0.3  # Lower alpha specifically for background
            )
            
            # 2. Create custom visualization for prediction
            pred_viz = self._custom_draw_sem_seg(
                img.copy(), 
                pred_sem_seg,
                alpha=0.7,  # Overall alpha
                background_alpha=0.3  # Lower alpha specifically for background
            )
            
            # Create a combined visualization
            combined_img = np.concatenate([gt_viz, pred_viz], axis=1)
            
            # Save visualizations
            basename = os.path.basename(file_name)
            
            # Save GT with High PPI using PIL
            gt_pil = Image.fromarray(gt_viz)
            gt_pil.save(os.path.join(self.vis_dir, f"{basename}_gt.png"), dpi=(300, 300))
            
            # Save Prediction with High PPI using PIL
            pred_pil = Image.fromarray(pred_viz)
            pred_pil.save(os.path.join(self.vis_dir, f"{basename}_pred.png"), dpi=(300, 300))

            # Save combined (keeping original logic for combined just in case, but changing to PNG for consistency)
            cv2.imwrite(
                os.path.join(self.vis_dir, f"{basename}_combined.jpg"),
                cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
            )
        
        logger.info(f"Generated {max_vis} visualizations")
    
    def visualize_worst_cases(self, num_samples=10):
        """Visualize the worst performing images for error analysis."""
        logger.info(f"Visualizing {num_samples} worst performing images...")
        
        if not self.ground_truth or not self.predictions or "per_image_iou" not in self.results:
            logger.warning("No data available for worst case visualization")
            return
        
        # Create directory for worst cases
        worst_cases_dir = os.path.join(self.vis_dir, "worst_cases")
        os.makedirs(worst_cases_dir, exist_ok=True)
        
        # Get IoU for each image
        image_ious = []
        for i, img_data in enumerate(self.results["per_image_iou"]["images"]):
            mean_iou = np.nanmean(list(img_data["iou"].values()))
            image_ious.append((i, mean_iou, img_data["file_name"]))
        
        # Sort by IoU (ascending) and take worst N
        image_ious.sort(key=lambda x: x[1])
        worst_images = image_ious[:num_samples]
        
        # Visualize each worst case
        logger.info("Worst performing images:")
        for rank, (idx, iou, file_name) in enumerate(worst_images):
            logger.info(f"{rank+1}. {os.path.basename(file_name)}: IoU = {iou:.4f}")
            
            # Get ground truth and prediction
            gt = self.ground_truth[idx]
            pred = self.predictions[idx]
            
            # Read image
            img = cv2.imread(file_name)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize ground truth mask to match prediction dimensions if needed
            gt_sem_seg = gt["sem_seg"]
            pred_sem_seg = pred["sem_seg"]
            if gt_sem_seg.shape != pred_sem_seg.shape:
                pred_h, pred_w = pred_sem_seg.shape
                gt_sem_seg = cv2.resize(gt_sem_seg, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
            
            # Draw segmentations with custom function (transparent background, no labels)
            gt_viz = self._custom_draw_sem_seg(img.copy(), gt_sem_seg, 
                                            alpha=0.7, background_alpha=0.3)
            pred_viz = self._custom_draw_sem_seg(img.copy(), pred_sem_seg, 
                                            alpha=0.7, background_alpha=0.3)
            
            # Create error visualization (keep as is)
            error_mask = (gt_sem_seg != pred_sem_seg)
            error_img = img.copy()
            error_img[error_mask] = [255, 0, 0]  # Mark errors in red
            
            # Create visualization with error highlighting
            alpha = 0.5
            error_vis = cv2.addWeighted(img, 1-alpha, error_img, alpha, 0)
            
            # Create combined visualization
            combined_img = np.concatenate([gt_viz, pred_viz, error_vis], axis=1)
            
            # Add title strip with IoU value
            h, w, _ = combined_img.shape
            label_img = np.ones((60, w, 3), dtype=np.uint8) * 255
            
            # Add text to label image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(label_img, f"GT", (w//6-20, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(label_img, f"Prediction", (w//2-60, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(label_img, f"Error Highlight", (5*w//6-80, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(label_img, f"IoU: {iou:.4f}", (w-200, 40), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Combine with label
            final_img = np.vstack([label_img, combined_img])
            
            # Save visualization
            basename = os.path.basename(file_name)
            output_path = os.path.join(worst_cases_dir, f"worst_{rank+1}_{basename}")
            cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Worst case visualizations saved to {worst_cases_dir}")

    def _plot_confusion_matrix(self):
        """Plot confusion matrix as a heatmap."""
        if "confusion_matrix" not in self.results:
            logger.warning("No confusion matrix available to plot")
            return
        
        conf_matrix = self.results["confusion_matrix"]["matrix"]
        conf_matrix_norm = self.results["confusion_matrix"]["matrix_normalized"]
        class_names = self.results["confusion_matrix"]["class_names"]
        
        # Plot absolute counts
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Counts)')
        plt.colorbar()
        
        # Add ticks and labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix_counts.png'), dpi=300)
        plt.close()
        
        # Plot normalized values (percentages)
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.title('Confusion Matrix (Normalized)')
        plt.colorbar()
        
        # Add ticks and labels
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = 0.5
        for i in range(conf_matrix_norm.shape[0]):
            for j in range(conf_matrix_norm.shape[1]):
                plt.text(j, i, f"{conf_matrix_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if conf_matrix_norm[i, j] > thresh else "black")
        
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix_normalized.png'), dpi=300)
        plt.close()

    def _plot_f1_scores(self):
        """Plot F1 scores, precision, and recall for each class."""
        if "f1_scores" not in self.results or "precision" not in self.results or "recall" not in self.results:
            logger.warning("No F1 scores available to plot")
            return
        
        f1_scores = self.results["f1_scores"]["per_class"]
        precision = self.results["precision"]["per_class"]
        recall = self.results["recall"]["per_class"]
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up the bar positions
        x = np.arange(len(self.class_names))
        width = 0.25
        
        # Create the bars
        rects1 = ax.bar(x - width, [precision[c] for c in self.class_names], width, label='Precision')
        rects2 = ax.bar(x, [recall[c] for c in self.class_names], width, label='Recall')
        rects3 = ax.bar(x + width, [f1_scores[c] for c in self.class_names], width, label='F1 Score')
        
        # Add labels and title
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 Score by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        
        # Add a horizontal grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add score values on top of bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.plots_dir, 'f1_scores.png'), dpi=300)
        plt.close()
        
        # Also create a table with metrics
        plt.figure(figsize=(10, 6))
        plt.axis('tight')
        plt.axis('off')
        
        table_data = []
        for c in self.class_names:
            table_data.append([
                c,
                f"{precision[c]:.4f}",
                f"{recall[c]:.4f}",
                f"{f1_scores[c]:.4f}"
            ])
        
        # Add macro average
        table_data.append([
            "Macro Avg",
            f"{self.results['precision']['macro_avg']:.4f}",
            f"{self.results['recall']['macro_avg']:.4f}",
            f"{self.results['f1_scores']['macro_avg']:.4f}"
        ])
        
        table = plt.table(
            cellText=table_data,
            colLabels=["Class", "Precision", "Recall", "F1 Score"],
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Precision, Recall, and F1 Score Metrics', pad=20)
        plt.tight_layout()
        
        # Save table
        plt.savefig(os.path.join(self.plots_dir, 'metrics_table.png'), dpi=300)
        plt.close()

    def _plot_per_image_iou(self):
        """Plot IoU distribution for each class."""
        if "per_image_iou" not in self.results:
            logger.warning("No per-image IoU available to plot")
            return
        
        # Extract IoU values for each class
        class_ious = defaultdict(list)
        for img_data in self.results["per_image_iou"]["images"]:
            for class_name, iou in img_data["iou"].items():
                if not np.isnan(iou):  # Ignore NaN values
                    class_ious[class_name].append(iou)
        
        # Plot IoU distributions as box plots
        plt.figure(figsize=(12, 6))
        
        # Prepare data for box plot
        data_to_plot = [class_ious[c] for c in self.class_names if class_ious[c]]
        labels = [c for c in self.class_names if class_ious[c]]
        
        # Create box plot
        box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
        
        # Customize box colors
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(plt.cm.tab10(i % 10))
        
        # Add class mean IoU values at the bottom
        for i, class_name in enumerate(labels):
            if class_name in self.results["per_image_iou"]["class_mean"]:
                mean_iou = self.results["per_image_iou"]["class_mean"][class_name]
                plt.text(i + 1, -0.05, f"Mean: {mean_iou:.2f}", 
                        ha='center', va='center', fontsize=8,
                        transform=plt.gca().get_xaxis_transform())
        
        # Add mean IoU line
        plt.axhline(y=self.results["per_image_iou"]["mean"], color='r', linestyle='-', 
                   label=f"Mean IoU: {self.results['per_image_iou']['mean']:.4f}")
        
        # Add labels and title
        plt.xlabel('Class')
        plt.ylabel('IoU')
        plt.title('IoU Distribution by Class')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.plots_dir, 'per_image_iou.png'), dpi=300)
        plt.close()
    
    def analyze_model_complexity(self):
        """Analyze model complexity and inference time."""
        logger.info("Analyzing model complexity and inference time...")
        
        try:
            # Calculate number of parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Calculate model size
            model_size_bytes = sum(p.nelement() * p.element_size() for p in self.model.parameters())
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            # Measure inference time on a sample of images
            inference_times = []
            
            sample_size = min(20, len(self.ground_truth))
            for i in range(sample_size):
                file_name = self.ground_truth[i]["file_name"]
                img = cv2.imread(file_name)
                if img is None:
                    continue
                
                # Prepare input
                height, width = img.shape[:2]
                img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                inputs = {"image": img_tensor, "height": height, "width": width}
                
                # Warm-up run
                with torch.no_grad():
                    _ = self.model([inputs])
                
                # Timed run
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model([inputs])
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            # Calculate statistics
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            
            # Store results
            self.results["model_analysis"] = {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "model_size_mb": float(model_size_mb),
                "avg_inference_time_seconds": float(avg_inference_time),
                "std_inference_time_seconds": float(std_inference_time),
                "fps": float(fps)
            }
            
            # Log results
            logger.info(f"Model size: {model_size_mb:.2f} MB")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Average inference time: {avg_inference_time:.4f}s (±{std_inference_time:.4f}s)")
            logger.info(f"Frames per second: {fps:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to analyze model complexity: {e}")
    
    def visualize_training_history(self):
        """Visualize training history if available from saved training logs."""
        history_path = os.path.join(os.path.dirname(self.output_dir), "history", "training_history.json")
        
        if not os.path.exists(history_path):
            logger.warning(f"No training history found at {history_path}")
            return
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Plot training loss and validation metrics
            plt.figure(figsize=(12, 6))
            
            # Plot training loss
            if 'loss' in history and 'iterations' in history:
                plt.plot(history['iterations'], history['loss'], 'b-', label='Training Loss')
            
            # Plot validation metrics
            if 'val_metrics' in history and history['val_metrics']:
                val_iters, val_metrics = zip(*history['val_metrics'])
                plt.plot(val_iters, val_metrics, 'r-', label='Validation mIoU')
            
            plt.xlabel('Iteration')
            plt.ylabel('Loss / Metric')
            plt.title('Training History')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), dpi=300)
            plt.close()
            
            # Plot per-class metrics if available
            if 'class_metrics' in history and history['class_metrics']:
                plt.figure(figsize=(12, 8))
                
                for class_name, metrics in history['class_metrics'].items():
                    if not metrics:
                        continue
                    
                    iters, values = zip(*metrics)
                    plt.plot(iters, values, '-', label=class_name)
                
                plt.xlabel('Iteration')
                plt.ylabel('IoU')
                plt.title('Per-Class Performance')
                plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save figure
                plt.savefig(os.path.join(self.plots_dir, 'per_class_metrics.png'), dpi=300)
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to visualize training history: {e}")
    
    def _save_results(self):
        """Save evaluation results to JSON file and generate summary report with metrics."""
        logger.info("Saving evaluation results...")
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if key == "confusion_matrix":
                cm_data = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        cm_data[k] = v.tolist()  # Convert numpy arrays to lists
                    else:
                        cm_data[k] = v
                results_json[key] = cm_data
            else:
                results_json[key] = value
        
        # Save to JSON file
        with open(os.path.join(self.metrics_dir, "evaluation_results.json"), "w") as f:
            json.dump(results_json, f, indent=2)
        
        # Generate summary report
        with open(os.path.join(self.output_dir, "evaluation_summary.md"), "w") as f:
            f.write(f"# {self.model_name} Evaluation Summary\n\n")
            
            f.write("## Dataset Information\n")
            f.write(f"- Dataset: {self.dataset_name}\n")
            f.write(f"- Number of images: {len(self.dataset)}\n")
            f.write(f"- Classes: {', '.join(self.class_names)}\n\n")
            
            f.write("## Standard Metrics\n")
            if "standard" in self.results:
                sem_seg_metrics = self.results["standard"].get("sem_seg", {})
                f.write(f"- mIoU: {sem_seg_metrics.get('mIoU', 'N/A'):.4f}\n")
                f.write(f"- fwIoU: {sem_seg_metrics.get('fwIoU', 'N/A'):.4f}\n")
                
                # Per-class IoU
                f.write("\n### Per-Class IoU\n")
                for class_name in self.class_names:
                    class_iou_key = f"IoU-{class_name}"
                    if class_iou_key in sem_seg_metrics:
                        f.write(f"- {class_name}: {sem_seg_metrics[class_iou_key]:.4f}\n")
            
            f.write("\n## F1 Scores\n")
            if "f1_scores" in self.results:
                f1_scores = self.results["f1_scores"]["per_class"]
                precision = self.results["precision"]["per_class"]
                recall = self.results["recall"]["per_class"]
                
                for class_name in self.class_names:
                    f.write(f"- {class_name}: F1={f1_scores[class_name]:.4f} (P={precision[class_name]:.4f}, R={recall[class_name]:.4f})\n")
                
                f.write(f"- Macro Average: F1={self.results['f1_scores']['macro_avg']:.4f} (P={self.results['precision']['macro_avg']:.4f}, R={self.results['recall']['macro_avg']:.4f})\n\n")
            
            # Add boundary analysis section
            if "boundary_analysis" in self.results:
                f.write("\n## Boundary Accuracy Analysis\n")
                f.write("Analysis of segmentation performance near boundaries:\n\n")
                
                for width, results in self.results["boundary_analysis"].items():
                    f.write(f"### Boundary Width: {width}px\n")
                    f.write(f"- Mean IoU: {results['mean']:.4f}\n")
                    f.write("- Per-class boundary IoU:\n")
                    
                    for class_name, iou in results["per_class"].items():
                        if iou > 0:
                            f.write(f"  - {class_name}: {iou:.4f}\n")
                
                f.write("\n![Boundary Analysis](plots/boundary_analysis.png)\n\n")
            
            # Add size sensitivity section
            if "size_sensitivity" in self.results:
                f.write("\n## Size Sensitivity Analysis\n")
                f.write("Analysis of performance on different image sizes:\n\n")
                
                for size, stats in self.results["size_sensitivity"].items():
                    if stats["count"] > 0:
                        f.write(f"- {size.capitalize()} images ({stats['count']}): ")
                        f.write(f"Mean IoU = {stats['mean_iou']:.4f}, ")
                        f.write(f"Median IoU = {stats['median_iou']:.4f}, ")
                        f.write(f"Std = {stats['std_iou']:.4f}\n")
                
                f.write("\n![Size Sensitivity Analysis](plots/size_sensitivity.png)\n\n")
            
            f.write("## Model Performance Analysis\n")
            if "model_analysis" in self.results:
                model_data = self.results["model_analysis"]
                f.write(f"- Model size: {model_data['model_size_mb']:.2f} MB\n")
                f.write(f"- Total parameters: {model_data['total_parameters']:,}\n")
                f.write(f"- Trainable parameters: {model_data['trainable_parameters']:,}\n")
                f.write(f"- Average inference time: {model_data['avg_inference_time_seconds']:.4f}s\n")
                f.write(f"- Frames per second: {model_data['fps']:.2f} FPS\n\n")
            
            # Include worst cases section
            f.write("\n## Error Analysis\n")
            f.write("The 10 worst performing images have been visualized in the `visualizations/worst_cases/` directory ")
            f.write("to help identify common failure patterns.\n\n")
            
            f.write("\n## Training History\n")
            f.write("![Training History](plots/training_history.png)\n\n")
            
            f.write("## Visualizations\n")
            f.write("See the `visualizations` directory for prediction examples.\n\n")
            
            f.write("## Plots\n")
            f.write("- Confusion matrix: `plots/confusion_matrix_counts.png` and `plots/confusion_matrix_normalized.png`\n")
            f.write("- F1 scores: `plots/f1_scores.png`\n")
            f.write("- Per-image IoU: `plots/per_image_iou.png`\n")
            f.write("- Training history: `plots/training_history.png`\n")

def setup_cfg(args):
    """Setup and return config."""
    # Create config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Merge configuration file
    cfg.merge_from_file(args.config_file)
    
    # Update from command line options
    cfg.merge_from_list(args.opts)
    
    # Set model weights
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--model-weights",
        help="Path to model weights (override config)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./evaluation",
        help="Base output directory for results"
    )
    
    parser.add_argument(
        "--model-name",
        default="mask2former_semantic",
        help="Model name for reports"
    )
    
    parser.add_argument(
        "--enhanced-eval",
        action="store_true",
        help="Enable enhanced evaluation metrics"
    )
    
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional options to override config"
    )
    
    args = parser.parse_args()
    
    # Register datasets
    datasets_semantic.register_semantic_datasets()
    
    # Setup base config
    base_cfg = setup_cfg(args)
    base_output_dir = args.output_dir
    
    # Define datasets to evaluate
    datasets_to_evaluate = ["corrosion_val", "corrosion_test"]
    logger.info(f"Scheduled to evaluate on: {datasets_to_evaluate}")

    for dataset_name in datasets_to_evaluate:
        logger.info("="*80)
        logger.info(f"🚀 STARTING EVALUATION FOR DATASET: {dataset_name}")
        logger.info("="*80)
        
        # 1. Create a dataset-specific output directory
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 2. Clone and update config for this specific dataset
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.DATASETS.TEST = (dataset_name,) 
        cfg.OUTPUT_DIR = dataset_output_dir
        cfg.freeze()
        
        # 3. Initialize and run evaluator
        evaluator = SemanticSegmentationEvaluator(
            cfg=cfg,
            dataset_name=dataset_name,
            output_dir=dataset_output_dir,
            model_name=args.model_name
        )
        
        # Run evaluation (conditionally with enhanced metrics if specified)
        if args.enhanced_eval:
            logger.info(f"Running enhanced evaluation metrics for {dataset_name}")
        
        results = evaluator.evaluate()
        
        logger.info(f"✅ Evaluation for {dataset_name} finished.")
    
    logger.info("🎉 All evaluations completed successfully!")
    logger.info(f"All results saved to base directory: {base_output_dir}")

if __name__ == "__main__":
    main()
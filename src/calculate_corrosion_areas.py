#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import time
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_util

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode

# Add Mask2Former path
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
sys.path.insert(0, MASK2FORMER_ROOT)

from mask2former import add_maskformer2_config, MaskFormerSemanticDatasetMapper

# Import dataset registration modules
import datasets_semantic
import datasets

# Setup logger
setup_logger()
logger = logging.getLogger("detectron2.corrosion_area_analysis")

# --- PLOTTING CONFIGURATION FOR Q1 JOURNAL ---
# Use a robust configuration that prefers Times New Roman but falls back to other Serifs
# to prevent "Font not found" warnings on Linux systems.
try:
    plt.rcParams['font.family'] = 'serif'
    # Matplotlib will try these in order. If Times New Roman isn't installed, it uses the next one.
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.autolayout'] = True
except Exception:
    logger.warning("Could not set font preferences. Using defaults.")

def rgb_to_bgr(rgb_color):
    """Convert RGB color tuple to BGR color tuple for OpenCV."""
    return (rgb_color[2], rgb_color[1], rgb_color[0])

# Define corrosion intensity colors (matches the dataset registration in datasets_semantic.py)
CORROSION_COLORS = {
    "Background": (0, 0, 0),   # Black
    "Fair": (255, 255, 0),     # Yellow
    "Poor": (255, 165, 0),     # Orange
    "Severe": (255, 0, 0)      # Red
}

CORROSION_CLASS_IDS = {
    "Background": 0,
    "Fair": 1,
    "Poor": 2,
    "Severe": 3
}

# --- HELPER FUNCTIONS FOR MATCHING ---

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
) -> List[Tuple[int, int]]:
    """Match GT and predicted instances using IoU with Hungarian algorithm."""
    if not gt_masks or not pred_masks:
        return []
    
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)
    iou_matrix = np.zeros((num_gt, num_pred))
    
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = calculate_iou(gt_mask, pred_mask)
    
    # Hungarian algorithm: Maximize IoU (minimize negative IoU)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    
    matches = []
    for gt_idx, pred_idx in zip(row_indices, col_indices):
        if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
            matches.append((gt_idx, pred_idx))
            
    return matches

def decode_rle_mask(segmentation, height: int, width: int) -> Optional[np.ndarray]:
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

class CorrosionAreaAnalyzer:
    """Analyzes corrosion areas within structural elements."""

    def __init__(
        self,
        semantic_cfg,
        instance_cfg,
        dataset_name: str,
        output_dir: str,
        visualization_dir: Optional[str] = None,
        min_confidence: float = 0.5,
        enhance_visualization: bool = False,
        simple_visualization: bool = False
    ):
        self.semantic_cfg = semantic_cfg
        self.instance_cfg = instance_cfg
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.enhance_visualization = enhance_visualization
        self.simple_visualization = simple_visualization
        
        # Create subdirectories
        self.data_dir = os.path.join(output_dir, "data")
        self.reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up visualization directory
        if visualization_dir:
            self.vis_dir = visualization_dir
        else:
            self.vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Create subdirectories for different visualization types
        self.overlap_vis_dir = os.path.join(self.vis_dir, "overlaps")
        self.element_vis_dir = os.path.join(self.vis_dir, "elements")
        self.corrosion_vis_dir = os.path.join(self.vis_dir, "corrosion")
        os.makedirs(self.overlap_vis_dir, exist_ok=True)
        os.makedirs(self.element_vis_dir, exist_ok=True)
        os.makedirs(self.corrosion_vis_dir, exist_ok=True)

        # Load metadata and dataset
        self.metadata = MetadataCatalog.get(dataset_name)
        self.dataset = DatasetCatalog.get(dataset_name)

        # Build and load semantic segmentation model
        logger.info("Loading semantic segmentation model...")
        self.semantic_model = build_model(semantic_cfg)
        self.semantic_model.eval()
        checkpointer = DetectionCheckpointer(self.semantic_model)
        checkpointer.load(semantic_cfg.MODEL.WEIGHTS)

        # Build and load instance segmentation model
        logger.info("Loading instance segmentation model...")
        self.instance_model = build_model(instance_cfg)
        self.instance_model.eval()
        checkpointer = DetectionCheckpointer(self.instance_model)
        checkpointer.load(instance_cfg.MODEL.WEIGHTS)

        # Get class names
        if hasattr(self.metadata, "stuff_classes"):
            self.corrosion_classes = self.metadata.stuff_classes
        else:
            self.corrosion_classes = [f"class_{i}" for i in range(semantic_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)]

        # Get structural element classes
        instance_metadata_name = "instance_val"
        if instance_metadata_name not in MetadataCatalog.list():
            logger.warning(f"Metadata for '{instance_metadata_name}' not found. Registering datasets.")
            datasets.register_datasets()
            
        instance_metadata = MetadataCatalog.get(instance_metadata_name)
        if hasattr(instance_metadata, "thing_classes"):
            self.structural_classes = instance_metadata.thing_classes
        else:
            self.structural_classes = ["Structural Element"]

        logger.info(f"Corrosion classes: {self.corrosion_classes}")
        logger.info(f"Structural classes: {self.structural_classes}")

        # Store results
        self.results = []
        
        # Store comparison data for the scatter plot
        self.comparison_data = defaultdict(list)
        
        # Store detailed comparison rows for Excel
        self.detailed_comparison_results = []
        
        # Store overall statistics
        self.overall_stats = {
            "total_images": 0,
            "total_elements": 0,
            "element_types": defaultdict(int),
            "corrosion_areas": defaultdict(int),
            "corrosion_ratios": defaultdict(list)
        }

    def analyze_dataset(self) -> List[Dict]:
        """Process all images in the dataset and analyze corrosion areas."""
        logger.info(f"Analyzing {len(self.dataset)} images from {self.dataset_name}...")
        
        self.results = []
        self.comparison_data = defaultdict(list)
        self.detailed_comparison_results = []
        
        # Infer instance dataset name
        if "val" in self.dataset_name:
            instance_ds_name = "instance_val"
        elif "test" in self.dataset_name:
            instance_ds_name = "instance_test"
        elif "train" in self.dataset_name:
            instance_ds_name = "instance_train"
        else:
            instance_ds_name = None
            
        instance_annotations_map = {}
        if instance_ds_name and instance_ds_name in DatasetCatalog.list():
            logger.info(f"Loading companion instance dataset: {instance_ds_name}")
            instance_dicts = DatasetCatalog.get(instance_ds_name)
            for d in instance_dicts:
                fname = os.path.basename(d['file_name'])
                instance_annotations_map[fname] = d.get('annotations', [])
        else:
            logger.warning(f"Could not find companion instance dataset for {self.dataset_name}. GT comparison will be skipped.")

        for idx, data_dict in enumerate(tqdm(self.dataset)):
            file_name = data_dict["file_name"]
            image_id = data_dict.get("image_id", os.path.basename(file_name))
            
            # Inject annotations if found in the map
            basename = os.path.basename(file_name)
            if basename in instance_annotations_map:
                data_dict['annotations'] = instance_annotations_map[basename]

            try:
                image_result = self.analyze_image(file_name, image_id, data_dict)
                if image_result["structural_elements"]:
                    self.results.append(image_result)
                    self.overall_stats["total_images"] += 1
                    self.overall_stats["total_elements"] += len(image_result["structural_elements"])
                    
                    for element in image_result["structural_elements"]:
                        self.overall_stats["element_types"][element["class_name"]] += 1
                        for corrosion_class, area_data in element["corrosion_areas"].items():
                            if corrosion_class != "Background":
                                self.overall_stats["corrosion_areas"][corrosion_class] += area_data["area_pixels"]
                                self.overall_stats["corrosion_ratios"][corrosion_class].append(area_data["area_ratio"])
                
            except Exception as e:
                logger.error(f"Error processing image {file_name}: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"Analysis complete. Processed {self.overall_stats['total_images']} images.")
        return self.results

    def analyze_image(self, file_name: str, image_id: str, data_dict: Dict = None) -> Dict:
        """Analyze a single image: Prediction AND Ground Truth Comparison."""
        image = read_image(file_name, format="RGB")
        if image is None:
            raise ValueError(f"Failed to read image: {file_name}")
            
        height, width = image.shape[:2]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. RUN PREDICTIONS
        instance_results = self._run_instance_model(image)
        semantic_results = self._run_semantic_model(image)
        
        # 2. VISUALIZATION
        pure_corrosion_vis = np.copy(image_bgr)
        corrosion_mask = np.zeros_like(image_bgr, dtype=np.uint8)
        
        for class_idx, class_name in enumerate(self.corrosion_classes):
            if class_name in CORROSION_COLORS and class_name != "Background":
                mask = (semantic_results == class_idx)
                rgb_color = CORROSION_COLORS[class_name]
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                corrosion_mask[mask] = bgr_color
                
        cv2.addWeighted(pure_corrosion_vis, 0.3, corrosion_mask, 0.7, 0, pure_corrosion_vis)
        pure_corrosion_path = os.path.join(self.corrosion_vis_dir, f"pure_corrosion_{os.path.basename(file_name)}")
        cv2.imwrite(pure_corrosion_path, pure_corrosion_vis)
        
        vis_image = np.copy(image_bgr)
        structural_elements = []
        pred_masks_list = []
        
        if "instances" in instance_results:
            instances = instance_results["instances"]
            if len(instances) > 0 and hasattr(instances, "pred_masks"):
                for i in range(len(instances)):
                    if hasattr(instances, "scores") and instances.scores[i].item() < self.min_confidence:
                        continue
                    
                    if hasattr(instances, "pred_classes"):
                        class_id = instances.pred_classes[i].item()
                        class_name = self.structural_classes[class_id]
                    else:
                        class_name = "Unknown"
                    
                    instance_mask = instances.pred_masks[i].cpu().numpy()
                    total_area = np.sum(instance_mask)
                    if total_area == 0: continue
                    
                    pred_masks_list.append(instance_mask)
                    
                    contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)
                    
                    y_positions, x_positions = np.where(instance_mask)
                    if len(y_positions) > 0:
                        cv2.putText(vis_image, f"{class_name}", (np.min(x_positions), max(0, np.min(y_positions) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    corrosion_areas = {}
                    for corrosion_idx, corrosion_name in enumerate(self.corrosion_classes):
                        if corrosion_name == "Background": continue
                        corr_mask = (semantic_results == corrosion_idx)
                        intersection = np.logical_and(instance_mask, corr_mask)
                        intersection_area = np.sum(intersection)
                        
                        corrosion_areas[corrosion_name] = {
                            "area_pixels": int(intersection_area),
                            "total_element_pixels": int(total_area),
                            "area_ratio": float(intersection_area / total_area)
                        }

                        if intersection_area > 0:
                            rgb_color = CORROSION_COLORS[corrosion_name]
                            bgr_color = rgb_to_bgr(rgb_color)
                            intersection_mask = np.zeros_like(image, dtype=np.uint8)
                            intersection_mask[intersection] = bgr_color
                            cv2.addWeighted(vis_image, 1.0, intersection_mask, 0.7, 0, vis_image)

                    structural_elements.append({
                        "element_id": i,
                        "class_name": class_name,
                        "total_area_pixels": int(total_area),
                        "corrosion_areas": corrosion_areas
                    })

        vis_path = os.path.join(self.vis_dir, f"{os.path.basename(file_name)}")
        cv2.imwrite(vis_path, vis_image)
        
        if len(structural_elements) > 0:
            h, w = image.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = pure_corrosion_vis
            comparison[:, w:] = vis_image
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
            cv2.putText(comparison, "Corrosion Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "Element Wise Corrosion", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            legend_height = 30 * (len(self.corrosion_classes) - 1)
            legend = np.ones((legend_height, w*2, 3), dtype=np.uint8) * 255
            legend_idx = 0
            for corrosion_name in self.corrosion_classes:
                if corrosion_name != "Background":
                    rgb_color = CORROSION_COLORS[corrosion_name]
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                    cv2.rectangle(legend, (10, legend_idx*30+5), (30, legend_idx*30+25), bgr_color, -1)
                    cv2.putText(legend, f"{corrosion_name} Corrosion", (40, legend_idx*30+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    legend_idx += 1
            
            comparison_with_legend = np.vstack([comparison, legend])
            cv2.imwrite(os.path.join(self.vis_dir, f"comparison_{os.path.basename(file_name)}"), comparison_with_legend)

        # 3. GROUND TRUTH INTEGRATION & MATCHING
        if data_dict and pred_masks_list:
            
            # A. Load Semantic GT
            semantic_gt_path = data_dict.get("sem_seg_file_name")
            semantic_gt = None
            if semantic_gt_path and os.path.exists(semantic_gt_path):
                semantic_gt = cv2.imread(semantic_gt_path, cv2.IMREAD_GRAYSCALE)
                if semantic_gt.shape != (height, width):
                    semantic_gt = cv2.resize(semantic_gt, (width, height), interpolation=cv2.INTER_NEAREST)

            # B. Load Instance GT
            gt_masks_list = []
            gt_annotations = data_dict.get("annotations", [])
            for ann in gt_annotations:
                mask = decode_rle_mask(ann.get('segmentation'), height, width)
                if mask is not None and mask.any():
                    gt_masks_list.append(mask)

            # C. Match Instances (Hungarian Algorithm on IoU)
            if gt_masks_list and semantic_gt is not None:
                matches = match_instances_by_iou(gt_masks_list, pred_masks_list, iou_threshold=0.5)
                
                # D. Compute Ratios for Matched Pairs
                for gt_idx, pred_idx in matches:
                    gt_mask = gt_masks_list[gt_idx]
                    if pred_idx >= len(structural_elements): continue
                    pred_data = structural_elements[pred_idx]
                    
                    gt_area_total = np.sum(gt_mask)
                    if gt_area_total == 0: continue

                    gt_ratios = {}
                    pred_ratios = {}
                    sum_gt_corrosion = 0
                    sum_pred_corrosion = 0
                    
                    # 1. Fair, Poor, Severe
                    for cls_name, cls_id in CORROSION_CLASS_IDS.items():
                        if cls_name == "Background": continue
                        
                        gt_corr_mask = (semantic_gt == cls_id)
                        gt_intersect = np.logical_and(gt_mask, gt_corr_mask).sum()
                        gt_r = gt_intersect / gt_area_total
                        gt_ratios[cls_name] = gt_r * 100
                        sum_gt_corrosion += gt_r
                        
                        pred_r = pred_data["corrosion_areas"].get(cls_name, {}).get("area_ratio", 0)
                        pred_ratios[cls_name] = pred_r * 100
                        sum_pred_corrosion += pred_r
                    
                    # 2. No Corrosion
                    gt_no_corr = max(0.0, 1.0 - sum_gt_corrosion) * 100
                    pred_no_corr = max(0.0, 1.0 - sum_pred_corrosion) * 100
                    
                    # Store Data for Plotting
                    self.comparison_data["No Corrosion"].append((gt_no_corr, pred_no_corr))
                    for cls_name in ["Fair", "Poor", "Severe"]:
                        self.comparison_data[cls_name].append((gt_ratios[cls_name], pred_ratios[cls_name]))

                    # Store detailed data for Excel
                    self.detailed_comparison_results.append({
                        "Image File": os.path.basename(file_name),
                        "Element Type": pred_data["class_name"],
                        "Element ID": pred_data["element_id"],
                        "IoU": calculate_iou(gt_mask, pred_masks_list[pred_idx]),
                        "GT No Corrosion (%)": gt_no_corr,
                        "Pred No Corrosion (%)": pred_no_corr,
                        "GT Fair (%)": gt_ratios.get("Fair", 0),
                        "Pred Fair (%)": pred_ratios.get("Fair", 0),
                        "GT Poor (%)": gt_ratios.get("Poor", 0),
                        "Pred Poor (%)": pred_ratios.get("Poor", 0),
                        "GT Severe (%)": gt_ratios.get("Severe", 0),
                        "Pred Severe (%)": pred_ratios.get("Severe", 0)
                    })

        return {
            "image_id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "structural_elements": structural_elements
        }

    def _run_instance_model(self, image):
        with torch.no_grad():
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image_tensor, "height": height, "width": width}
            predictions = self.instance_model([inputs])[0]
            return predictions

    def _run_semantic_model(self, image):
        with torch.no_grad():
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image_tensor, "height": height, "width": width}
            outputs = self.semantic_model([inputs])[0]
            if "sem_seg" in outputs:
                return outputs["sem_seg"].argmax(dim=0).cpu().numpy()
            else:
                return np.zeros((height, width), dtype=np.int64)

    def generate_summary(self, detailed_tables=False):
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        total_images = self.overall_stats["total_images"]
        total_elements = self.overall_stats["total_elements"]
        element_stats = defaultdict(lambda: {"count": 0, "total_area": 0, "corrosion_areas": {c: 0 for c in self.corrosion_classes}})
        
        for img in self.results:
            for element in img["structural_elements"]:
                class_name = element["class_name"]
                stats = element_stats[class_name]
                stats["count"] += 1
                stats["total_area"] += element["total_area_pixels"]
                for corrosion_class, area_data in element["corrosion_areas"].items():
                    if corrosion_class != "Background":
                        stats["corrosion_areas"][corrosion_class] += area_data["area_pixels"]

        for class_name, stats in element_stats.items():
            for corrosion_class in self.corrosion_classes:
                if corrosion_class != "Background":
                    area = stats["corrosion_areas"].get(corrosion_class, 0)
                    stats["corrosion_areas"][corrosion_class] = {
                        "area_pixels": area,
                        "percentage": (area / stats["total_area"]) * 100 if stats["total_area"] > 0 else 0
                    }

        image_stats = {}
        for img in self.results:
            image_id = img["image_id"]
            image_stats[image_id] = {
                "file_name": img["file_name"],
                "num_elements": len(img["structural_elements"]),
                "element_types": defaultdict(int),
                "corrosion_areas": defaultdict(int),
                "total_area": 0
            }
            for element in img["structural_elements"]:
                class_name = element["class_name"]
                image_stats[image_id]["element_types"][class_name] += 1
                image_stats[image_id]["total_area"] += element["total_area_pixels"]
                for corrosion_class, area_data in element["corrosion_areas"].items():
                    if corrosion_class != "Background":
                        image_stats[image_id]["corrosion_areas"][corrosion_class] += area_data["area_pixels"]
        
        for image_id, stats in image_stats.items():
            if stats["total_area"] > 0:
                for corrosion_class, area in stats["corrosion_areas"].items():
                    stats["corrosion_areas"][corrosion_class] = {"area_pixels": area, "percentage": (area / stats["total_area"]) * 100}

        summary = {"total_images": total_images, "total_elements": total_elements, "element_stats": element_stats, "image_stats": image_stats}
        
        with open(os.path.join(self.data_dir, "area_analysis_summary.json"), "w") as f:
            json_summary = {
                "total_images": summary["total_images"],
                "total_elements": summary["total_elements"],
                "element_stats": {k: dict(v) for k, v in summary["element_stats"].items()},
                "image_stats": {k: {
                    "file_name": v["file_name"],
                    "num_elements": v["num_elements"],
                    "element_types": dict(v["element_types"]),
                    "corrosion_areas": {c: (dict(a) if isinstance(a, dict) else a) for c, a in v["corrosion_areas"].items()},
                    "total_area": v["total_area"]
                } for k, v in summary["image_stats"].items()}
            }
            json.dump(json_summary, f, indent=2)

        if detailed_tables:
            self._generate_detailed_tables(summary)
        self._generate_markdown_report(summary)
        self._generate_summary_plots(summary)
        self._generate_comparison_plot()

        return summary
    def _generate_comparison_plot(self):
        """Generate the specific GT vs Prediction Scatter Plot for Q1 Journal."""
        if not self.comparison_data:
            logger.warning("No comparison data available for plotting.")
            return

        # --- CONFIGURATION FOR SIZES ---
        scatter_marker_size = 300  # Size of markers in the plot
        legend_marker_size = 150   # Size of markers in the legend
        # -------------------------------

        # Figure size
        plt.figure(figsize=(10, 10), dpi=600)
    
        plot_colors = {
            "No Corrosion": "#00CC00",  # Vivid Lime Green
            "Fair": "#FFFF00",          # Pure, Bright Yellow
            "Poor": "#FF8C00",          # Dark Orange
            "Severe": "#FF0000"         # Pure Red
        }

        # (c) Define distinct markers for each class
        plot_markers = {
            "No Corrosion": "o",  # Circle
            "Fair": "s",          # Square
            "Poor": "D",          # Diamond
            "Severe": "v"         # Inverted Triangle
        }
        
        classes = ["No Corrosion", "Fair", "Poor", "Severe"]
        
        # (a) Extend line all the way (-5 to 105 to match limits)
        plt.plot([-5, 105], [-5, 105], 'k--', linewidth=2)
        
        # --- NEW: Arrays to store all points for R2 calculation ---
        all_gt_vals = []
        all_pred_vals = []
        # ----------------------------------------------------------

        for cls_name in classes:
            data_points = self.comparison_data.get(cls_name, [])
            if not data_points: continue
                
            gt_vals = [pt[0] for pt in data_points]
            pred_vals = [pt[1] for pt in data_points]
            
            # --- NEW: Collect data ---
            all_gt_vals.extend(gt_vals)
            all_pred_vals.extend(pred_vals)
            # -------------------------
            
            plt.scatter(
                gt_vals, 
                pred_vals, 
                c=plot_colors[cls_name], 
                label=cls_name, 
                alpha=0.8,
                edgecolors='black',      # (d) Black border
                linewidth=2.0,           # (d) Thicker border to make color pop
                s=scatter_marker_size,   # (b) Bigger markers (using variable)
                marker=plot_markers[cls_name] # (c) Change marker type per class
            )

        # --- NEW: Calculate and Report R2 ---
        if all_gt_vals:
            gt_arr = np.array(all_gt_vals)
            pred_arr = np.array(all_pred_vals)
            
            # Calculate R^2 manually (1 - SS_res / SS_tot)
            ss_res = np.sum((gt_arr - pred_arr) ** 2)
            ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
            
            r2 = 0.0
            if ss_tot != 0:
                r2 = 1 - (ss_res / ss_tot)
            
            # Report to logger
            logger.info(f"Comparison Plot Statistics: R^2 = {r2:.4f}")
            
            # Report on Plot (Bottom Right to avoid Legend)
            plt.text(0.95, 0.05, f"$R^2 = {r2:.2f}$", 
                     transform=plt.gca().transAxes, 
                     fontsize=24, 
                     family='serif',
                     horizontalalignment='right', 
                     verticalalignment='bottom')
        # ------------------------------------

        # Define separate properties for labels and ticks
        label_font = {'family': 'serif', 'size': 20} 
        tick_font = {'family': 'serif', 'size': 18}  
        
        plt.xlabel("Ground Truth Area Ratio (%)", fontdict=label_font)
        plt.ylabel("Predicted Area Ratio (%)", fontdict=label_font)
        
        # Set limits
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        
        # Apply font settings to ticks
        plt.xticks(**tick_font)
        plt.yticks(**tick_font)
        
        plt.grid(False)
        
        # (e) Legend inside, no border
        leg = plt.legend(prop={'family': 'serif', 'size': 16}, loc='upper left', frameon=False)

        # Manually resize the markers in the legend
        for handle in leg.legend_handles:
            handle.set_sizes([legend_marker_size])
        
        output_path = os.path.join(self.reports_dir, "gt_vs_pred_corrosion_ratios.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison plot saved to: {output_path}")
        
    def _generate_detailed_tables(self, summary):
        element_table = []
        for img_idx, img in enumerate(self.results):
            for element in img["structural_elements"]:
                fair_data = element["corrosion_areas"].get("Fair", {"area_pixels": 0, "area_ratio": 0})
                poor_data = element["corrosion_areas"].get("Poor", {"area_pixels": 0, "area_ratio": 0})
                severe_data = element["corrosion_areas"].get("Severe", {"area_pixels": 0, "area_ratio": 0})
                element_table.append([
                    img_idx, os.path.basename(img["file_name"]), element["element_id"], element["class_name"],
                    element["total_area_pixels"],
                    fair_data["area_pixels"], f"{fair_data['area_ratio']*100:.2f}%",
                    poor_data["area_pixels"], f"{poor_data['area_ratio']*100:.2f}%",
                    severe_data["area_pixels"], f"{severe_data['area_ratio']*100:.2f}%"
                ])
        
        with open(os.path.join(self.data_dir, "element_level_corrosion.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Image File", "Element ID", "Element Type", "Total Area (px)", "Fair Area (px)", "Fair (%)", "Poor Area (px)", "Poor (%)", "Severe Area (px)", "Severe (%)"])
            writer.writerows(element_table)
            
        image_table = []
        for img_idx, img in enumerate(self.results):
            file_name = os.path.basename(img["file_name"])
            total_area = sum(e["total_area_pixels"] for e in img["structural_elements"])
            fair_area = sum(e["corrosion_areas"].get("Fair", {}).get("area_pixels", 0) for e in img["structural_elements"])
            poor_area = sum(e["corrosion_areas"].get("Poor", {}).get("area_pixels", 0) for e in img["structural_elements"])
            severe_area = sum(e["corrosion_areas"].get("Severe", {}).get("area_pixels", 0) for e in img["structural_elements"])
            fair_pct = (fair_area / total_area) * 100 if total_area > 0 else 0
            poor_pct = (poor_area / total_area) * 100 if total_area > 0 else 0
            severe_pct = (severe_area / total_area) * 100 if total_area > 0 else 0
            image_table.append([img_idx, file_name, len(img["structural_elements"]), total_area, fair_area, f"{fair_pct:.2f}%", poor_area, f"{poor_pct:.2f}%", severe_area, f"{severe_pct:.2f}%"])
        
        with open(os.path.join(self.data_dir, "image_level_corrosion.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Image File", "Num Elements", "Total Area (px)", "Fair Area (px)", "Fair (%)", "Poor Area (px)", "Poor (%)", "Severe Area (px)", "Severe (%)"])
            writer.writerows(image_table)

    def _generate_markdown_report(self, summary):
        with open(os.path.join(self.reports_dir, "area_analysis_summary.md"), "w") as f:
            f.write(f"# Corrosion Area Analysis Summary ({self.dataset_name})\n\n")
            f.write("## Overview\n")
            f.write(f"- Total images analyzed: {summary['total_images']}\n")
            f.write(f"- Total structural elements detected: {summary['total_elements']}\n\n")
            f.write("## Structural Element Statistics\n\n")
            f.write("| Structural Element | Count | Total Area (px) | Fair (%) | Poor (%) | Severe (%) |\n")
            f.write("|-------------------|-------|----------------|----------|----------|------------|\n")
            for class_name, stats in summary["element_stats"].items():
                fair_pct = stats["corrosion_areas"]["Fair"]["percentage"] if "Fair" in stats["corrosion_areas"] else 0
                poor_pct = stats["corrosion_areas"]["Poor"]["percentage"] if "Poor" in stats["corrosion_areas"] else 0
                severe_pct = stats["corrosion_areas"]["Severe"]["percentage"] if "Severe" in stats["corrosion_areas"] else 0
                f.write(f"| {class_name} | {stats['count']} | {stats['total_area']} | {fair_pct:.2f}% | {poor_pct:.2f}% | {severe_pct:.2f}% |\n")

    def _generate_summary_plots(self, summary):
        # 1. Corrosion distribution by structural element type
        plt.figure(figsize=(12, 8))
        element_types = list(summary["element_stats"].keys())
        corrosion_types = ["Fair", "Poor", "Severe"]
        x = np.arange(len(element_types))
        width = 0.25

        for i, corrosion_type in enumerate(corrosion_types):
            percentages = []
            for element_type in element_types:
                stats = summary["element_stats"][element_type]
                if corrosion_type in stats["corrosion_areas"] and isinstance(stats["corrosion_areas"][corrosion_type], dict):
                    percentages.append(stats["corrosion_areas"][corrosion_type]["percentage"])
                else:
                    percentages.append(0)
            plt.bar(x + (i - 1) * width, percentages, width, label=corrosion_type, color=tuple(c/255 for c in CORROSION_COLORS[corrosion_type]))
        
        plt.xlabel('Structural Element Type')
        plt.ylabel('Percentage (%)')
        plt.title('Corrosion Distribution by Structural Element Type')
        plt.xticks(x, element_types, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "corrosion_distribution.png"), dpi=300)
        plt.close()

    def save_detailed_results(self, output_format="all"):
        with open(os.path.join(self.data_dir, "detailed_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        if output_format in ["csv", "all"]:
            self._save_as_csv()
        if output_format in ["excel", "all"]:
            try:
                import pandas as pd
                self._save_as_excel()
            except ImportError:
                logger.warning("pandas not installed, skipping Excel output")

    def _save_as_csv(self):
        # Implementation skipped for brevity (same as previous)
        pass

    def _save_as_excel(self):
        import pandas as pd
        element_data = []
        for img in self.results:
            for element in img["structural_elements"]:
                fair_pct = element["corrosion_areas"]["Fair"]["area_ratio"] * 100
                poor_pct = element["corrosion_areas"]["Poor"]["area_ratio"] * 100
                severe_pct = element["corrosion_areas"]["Severe"]["area_ratio"] * 100
                element_data.append({
                    "Image ID": img["image_id"],
                    "Image File": img["file_name"],
                    "Element ID": element["element_id"],
                    "Element Type": element["class_name"],
                    "Total Area (px)": element["total_area_pixels"],
                    "Fair (%)": fair_pct,
                    "Poor (%)": poor_pct,
                    "Severe (%)": severe_pct
                })
        
        image_data = []
        for img in self.results:
            total_area = sum(e["total_area_pixels"] for e in img["structural_elements"])
            fair_area = sum(e["corrosion_areas"].get("Fair", {}).get("area_pixels", 0) for e in img["structural_elements"])
            poor_area = sum(e["corrosion_areas"].get("Poor", {}).get("area_pixels", 0) for e in img["structural_elements"])
            severe_area = sum(e["corrosion_areas"].get("Severe", {}).get("area_pixels", 0) for e in img["structural_elements"])
            image_data.append({
                "Image ID": img["image_id"],
                "Image File": img["file_name"],
                "Total Elements": len(img["structural_elements"]),
                "Total Area (px)": total_area,
                "Fair (%)": (fair_area / total_area) * 100 if total_area > 0 else 0,
                "Poor (%)": (poor_area / total_area) * 100 if total_area > 0 else 0,
                "Severe (%)": (severe_area / total_area) * 100 if total_area > 0 else 0
            })
        
        writer = pd.ExcelWriter(os.path.join(self.data_dir, "corrosion_analysis.xlsx"), engine="xlsxwriter")
        pd.DataFrame(element_data).to_excel(writer, sheet_name="Element Level Data", index=False)
        pd.DataFrame(image_data).to_excel(writer, sheet_name="Image Level Data", index=False)
        
        if self.detailed_comparison_results:
            pd.DataFrame(self.detailed_comparison_results).to_excel(writer, sheet_name="GT vs Pred Comparison", index=False)
        
        writer.close()
        logger.info(f"Saved Excel report to {os.path.join(self.data_dir, 'corrosion_analysis.xlsx')}")

def setup_semantic_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.semantic_config)
    if args.semantic_weights: cfg.MODEL.WEIGHTS = args.semantic_weights
    cfg.merge_from_list(args.opts)
    return cfg

def setup_instance_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    if not hasattr(cfg, "TEST"):
        from fvcore.common.config import CfgNode as CN
        cfg.TEST = CN()
    if not hasattr(cfg.TEST, "SEMANTIC_ON"): cfg.TEST.SEMANTIC_ON = False
    if not hasattr(cfg.TEST, "INSTANCE_ON"): cfg.TEST.INSTANCE_ON = True
    if not hasattr(cfg.TEST, "PANOPTIC_ON"): cfg.TEST.PANOPTIC_ON = False
    cfg.SOLVER.BETAS = [0.9, 0.999]
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.instance_config)
    if args.instance_weights: cfg.MODEL.WEIGHTS = args.instance_weights
    cfg.merge_from_list(args.opts)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Calculate corrosion areas within structural elements")
    parser.add_argument("--semantic-config", required=True, help="Path to semantic segmentation config file")
    parser.add_argument("--semantic-weights", help="Path to semantic segmentation model weights")
    parser.add_argument("--instance-config", required=True, help="Path to instance segmentation config file")
    parser.add_argument("--instance-weights", help="Path to instance segmentation model weights")
    parser.add_argument("--output-dir", default="./corrosion_area_analysis", help="Base output directory for results")
    parser.add_argument("--output-format", choices=["json", "csv", "excel", "all"], default="all", help="Output format for results")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence threshold for instance detection")
    parser.add_argument("--detailed-tables", action="store_true", help="Generate detailed tables")
    parser.add_argument("--enhance-visualization", action="store_true", help="Generate enhanced visualizations")
    parser.add_argument("--simple-visualization", action="store_true", help="Use simplified visualization")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional options")
    
    args = parser.parse_args()
    
    datasets_semantic.register_semantic_datasets()
    datasets.register_datasets()
    
    base_semantic_cfg = setup_semantic_cfg(args)
    base_instance_cfg = setup_instance_cfg(args)
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    datasets_to_evaluate = ["corrosion_val", "corrosion_test"]
    logger.info(f"Scheduled to evaluate on: {datasets_to_evaluate}")

    for dataset_name in datasets_to_evaluate:
        logger.info("="*80)
        logger.info(f"🚀 STARTING AREA ANALYSIS FOR DATASET: {dataset_name}")
        logger.info("="*80)

        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        dataset_vis_dir = os.path.join(dataset_output_dir, "visualizations")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        analyzer = CorrosionAreaAnalyzer(
            semantic_cfg=base_semantic_cfg.clone(),
            instance_cfg=base_instance_cfg.clone(),
            dataset_name=dataset_name,
            output_dir=dataset_output_dir,
            visualization_dir=dataset_vis_dir, 
            min_confidence=args.min_confidence,
            enhance_visualization=args.enhance_visualization,
            simple_visualization=args.simple_visualization
        )
        
        analyzer.analyze_dataset()
        analyzer.generate_summary(detailed_tables=args.detailed_tables)
        analyzer.save_detailed_results(output_format=args.output_format)
    
    logger.info("🎉 All area analyses completed successfully!")
    logger.info(f"All results saved to base directory: {base_output_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Mask2Former Comprehensive Evaluation Script
"""
import sys
import os
import logging
import json
import time
import argparse
import re
from typing import Dict, List, Any
import numpy as np
import cv2
import torch
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from pycocotools import mask as mask_util

# --- Project Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
if os.path.exists(MASK2FORMER_ROOT):
    sys.path.insert(0, MASK2FORMER_ROOT)

# --- Detectron2 & Mask2Former Imports ---
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances, Boxes, BoxMode
from mask2former import add_maskformer2_config
import datasets

# --- Setup Logging and Plotting Style ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'figure.dpi': 120, 'savefig.dpi': 300, 'figure.figsize': (14, 8),
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'figure.titlesize': 'large',
    'figure.titleweight': 'bold', 'legend.fontsize': 'medium'
})

def safe_save_figure(fig, filepath):
    """Saves a matplotlib figure, closing it afterward."""
    try:
        # pad_inches=0 removes extra white border since we have no titles
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
        logger.info(f"✅ Saved plot: {os.path.basename(filepath)}")
    except Exception as e:
        logger.error(f"❌ Failed to save figure {filepath}: {e}")
    finally:
        plt.close(fig)

class ComprehensiveEvaluator:
    def __init__(self, cfg, output_dir: str, training_log: str):
        self.cfg = cfg
        self.output_dir = output_dir
        self.training_log = training_log
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.dataset = DatasetCatalog.get(self.dataset_name)
        self.metadata = MetadataCatalog.get(self.dataset_name)
        self.class_names = self.metadata.thing_classes
        self.coco_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id
        
        self.plots_dir = os.path.join(output_dir, "plots")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        logger.info(f"Evaluator initialized. Results will be saved to: {self.output_dir}")

    def run(self):
        """Execute the full evaluation pipeline."""
        logger.info(f"🚀 Starting full evaluation pipeline for {self.dataset_name}...")
        
        evaluator = COCOEvaluator(self.dataset_name, tasks=("segm",), output_dir=self.output_dir)
        data_loader = build_detection_test_loader(self.cfg, self.dataset_name, mapper=DatasetMapper(self.cfg, is_train=False))
        
        logger.info("📊 Running inference and evaluation...")
        results = inference_on_dataset(DefaultPredictor(self.cfg).model, data_loader, evaluator)
        coco_results = results.get("segm", {})

        predictions_file = os.path.join(self.output_dir, "coco_instances_results.json")
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)

        metrics = self._calculate_pr_and_confusion_matrix(predictions)
        self._generate_all_plots(coco_results, metrics)
        self._generate_all_visualizations(predictions)
        self._generate_summary_report(metrics, coco_results)
        
        logger.info(f"✅ Evaluation pipeline for {self.dataset_name} finished successfully.")

    def _calculate_pr_and_confusion_matrix(self, predictions: List[Dict]) -> Dict:
        logger.info("🧮 Calculating Precision, Recall, F1, and Confusion Matrix...")
        
        gt_data = {item['image_id']: item for item in self.dataset}
        preds_by_image = defaultdict(list)
        for p in predictions: preds_by_image[p['image_id']].append(p)
        
        num_classes = len(self.class_names)
        # Matrix size is num_classes x (num_classes + 1) to include an "Undetected" column
        conf_matrix = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
        iou_thresh = 0.5; score_thresh = 0.5
        undetected_idx = num_classes

        for gt_info in self.dataset:
            image_id = gt_info['image_id']
            h, w = gt_info['height'], gt_info['width']
            gt_anns = gt_info.get("annotations", [])
            pred_anns = [p for p in preds_by_image.get(image_id, []) if p['score'] >= score_thresh]

            if not gt_anns: continue

            gt_masks = [mask_util.decode(mask_util.merge(mask_util.frPyObjects(ann['segmentation'], h, w))) for ann in gt_anns]
            gt_cats = [ann['category_id'] for ann in gt_anns]
            
            pred_masks = [mask_util.decode(p['segmentation']) for p in pred_anns]
            pred_cats = [self.coco_id_to_contiguous_id[p['category_id']] for p in pred_anns]
            
            gt_matched = [False] * len(gt_masks)
            pred_matched = [False] * len(pred_masks)
            
            if len(gt_masks) > 0 and len(pred_masks) > 0:
                iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
                for i, g_mask in enumerate(gt_masks):
                    for j, p_mask in enumerate(pred_masks):
                        intersection = np.sum(np.logical_and(g_mask, p_mask))
                        union = np.sum(np.logical_or(g_mask, p_mask))
                        iou = intersection / union if union > 0 else 0
                        iou_matrix[i, j] = iou
                
                # Match each ground truth to the best available prediction
                for i, g_cat in enumerate(gt_cats):
                    best_iou_for_gt = -1
                    best_pred_idx = -1
                    for j in range(len(pred_masks)):
                        if not pred_matched[j] and iou_matrix[i, j] > best_iou_for_gt:
                            best_iou_for_gt = iou_matrix[i, j]
                            best_pred_idx = j
                    
                    if best_iou_for_gt >= iou_thresh:
                        p_cat = pred_cats[best_pred_idx]
                        conf_matrix[g_cat, p_cat] += 1
                        gt_matched[i] = True
                        pred_matched[best_pred_idx] = True

            # Any unmatched ground truth is a False Negative (Undetected)
            for i, matched in enumerate(gt_matched):
                if not matched:
                    conf_matrix[gt_cats[i], undetected_idx] += 1

        # Derive TP, FP, FN from the complete confusion matrix
        metrics = {}
        for i, name in enumerate(self.class_names):
            tp = conf_matrix[i, i]
            # FN is the sum of the GT row minus the TP cell (misclassifications + undetected)
            fn = conf_matrix[i, :].sum() - tp
            # FP is the sum of the prediction column minus the TP cell (misclassifications from other classes)
            fp = conf_matrix[:, i].sum() - tp

            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0 # This is now the true recall
            f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0.0
            metrics[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}
        
        metrics['confusion_matrix'] = conf_matrix
        return metrics
    
    def _generate_all_plots(self, coco_results, metrics):
        logger.info("🎨 Generating all performance plots...")
        self._plot_coco_summary(coco_results)
        self._plot_per_class_performance(metrics)
        self._plot_confusion_matrix(metrics['confusion_matrix'])
        self._plot_training_curves() 
        self._plot_score_distribution(os.path.join(self.output_dir, "coco_instances_results.json"))

    def _plot_coco_summary(self, coco_results):
        metrics_data = {k: v for k, v in coco_results.items() if "AP" in k and "-" not in k}
        if not metrics_data: return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(metrics_data.keys(), metrics_data.values(), 
                     color=sns.color_palette("viridis", len(metrics_data)))
        ax.set_ylim(0, 100)
        ax.set_title("COCO Evaluation Metrics Summary", fontsize=16, fontweight='bold')
        ax.set_ylabel("Score (%)", fontsize=14)
        ax.set_xlabel("Metrics", fontsize=14)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
        
        safe_save_figure(fig, os.path.join(self.plots_dir, "coco_metrics_summary.png"))

    def _plot_per_class_performance(self, metrics):
        plot_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        if not plot_metrics: return
        
        df = pd.DataFrame(plot_metrics).T
        df = df[['precision', 'recall', 'f1']].astype(float).sort_values('f1', ascending=False) * 100
        
        fig, ax = plt.subplots(figsize=(12, 8))
        df.plot(kind='bar', ax=ax, colormap='cividis')
        ax.set_title("Per-Class Precision, Recall, and F1-Score (IoU > 0.5)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Score (%)", fontsize=14)
        ax.set_xlabel("Classes", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title="Metric", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        
        safe_save_figure(fig, os.path.join(self.plots_dir, "per_class_performance.png"))

    def _plot_training_curves(self):
        """Improved log parsing for detectron2 format - searches for AP (which is mAP)"""
        if not os.path.exists(self.training_log):
            logger.warning(f"Training log '{self.training_log}' not found. Skipping training curve plots.")
            return

        logger.info(f"📈 Parsing training data from {self.training_log}...")
        data = defaultdict(list)
        
        # Pattern matching for detectron2 log format
        loss_pattern = r'.*iter:\s*(\d+).*total_loss:\s*([\d.]+).*lr:\s*([\d.e-]+)'
        iter_pattern = r'.*iter:\s*(\d+)'
        
        current_iter = None
        
        with open(self.training_log, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Parse training loss, iteration, and learning rate
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                iter_val = int(loss_match.group(1))
                total_loss = float(loss_match.group(2))
                lr = float(loss_match.group(3))
                
                data['iter'].append(iter_val)
                data['total_loss'].append(total_loss)
                data['lr'].append(lr)
                current_iter = iter_val
                continue
            
            # Parse AP values from evaluation results (AP is mAP in training logs)
            # Only record ONE AP value per iteration to avoid duplicates
            if current_iter is not None and ('AP' in line or 'copypaste:' in line):
                # Check if we already have an AP value for this iteration
                if current_iter in data['val_iter']:
                    continue  # Skip if we already recorded AP for this iteration
                
                try:
                    ap_val = None
                    
                    # Format 1: "copypaste: 0.45123,0.67456,..." (comma separated values)
                    # This is the most reliable format - prioritize it
                    if 'copypaste:' in line and ',' in line:
                        numbers_match = re.search(r'copypaste:\s*([\d.,]+)', line)
                        if numbers_match:
                            numbers_str = numbers_match.group(1)
                            numbers = [float(x) for x in numbers_str.split(',') if x.strip()]
                            if len(numbers) >= 1:  # First number is AP (mAP)
                                ap_val = numbers[0]
                    
                    # Format 2: "| 45.123| 67.456| 48.789| ..." (table format)
                    elif '|' in line and re.search(r'\|\s*([\d.]+)\s*\|', line):
                        # Only take the FIRST value in the table (which should be overall AP)
                        ap_match = re.search(r'\|\s*([\d.]+)\s*\|', line)
                        if ap_match:
                            ap_val = float(ap_match.group(1))
                    
                    # Format 3: Look for "AP:" followed by a number (but not AP50, AP75, etc.)
                    elif re.search(r'\bAP:\s*([\d.]+)', line) and 'AP50' not in line and 'AP75' not in line:
                        ap_match = re.search(r'\bAP:\s*([\d.]+)', line)
                        if ap_match:
                            ap_val = float(ap_match.group(1))
                    
                    # Only record if we found a valid AP value
                    if ap_val is not None:
                        # Convert to percentage if needed
                        if ap_val <= 1.0:
                            ap_val *= 100
                        
                        data['val_iter'].append(current_iter)
                        data['val_ap'].append(ap_val)
                        continue
                            
                except (ValueError, IndexError, AttributeError):
                    continue
            
            # Look for iteration numbers in lines without full loss info
            if not loss_match:
                iter_match = re.search(iter_pattern, line)
                if iter_match:
                    current_iter = int(iter_match.group(1))

        if not data['iter']:
            logger.warning("❌ Could not parse training data from log. Check log format.")
            return
            
        logger.info(f"✅ Parsed {len(data['iter'])} training iterations and {len(data.get('val_ap', []))} validation points")
        
        # Plot 1: Training Loss
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(data['total_loss']) > 1:
            # Raw loss
            ax.plot(data['iter'], data['total_loss'], alpha=0.3, color='blue', label='Training Loss (Raw)')
            # Smoothed loss
            smoothed_loss = pd.Series(data['total_loss']).ewm(span=min(50, len(data['total_loss'])//4)).mean()
            ax.plot(data['iter'], smoothed_loss, color='blue', linewidth=2, label='Training Loss (Smoothed)')
        
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Training Loss", fontsize=14)
        ax.set_title("Training Loss Curve", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        safe_save_figure(fig, os.path.join(self.plots_dir, "training_loss_curve.png"))
        
        # Plot 2: Validation mAP
        if data.get('val_ap') and len(data['val_ap']) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['val_iter'], data['val_ap'], 'o-', color='green', 
                   linewidth=2, markersize=6, label="Validation mAP")
            ax.set_xlabel("Iteration", fontsize=14)
            ax.set_ylabel("Validation mAP (%)" if max(data['val_ap']) > 1 else "Validation mAP", fontsize=14)
            ax.set_title("Validation mAP Curve", fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0, top=max(data['val_ap']) * 1.1 if data['val_ap'] else 1)
            safe_save_figure(fig, os.path.join(self.plots_dir, "validation_map_curve.png"))
        else:
            logger.warning("⚠️ No validation mAP data found in training log")
        
        # Plot 3: Learning Rate Schedule
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['iter'], data['lr'], color='red', linewidth=2, label="Learning Rate")
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Learning Rate", fontsize=14)
        ax.set_yscale('log')
        ax.set_title("Learning Rate Schedule", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        safe_save_figure(fig, os.path.join(self.plots_dir, "learning_rate_curve.png"))

    def _plot_confusion_matrix(self, matrix):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        labels = self.class_names + ["Undetected"]
        
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(matrix.astype('float'), row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums!=0)
        
        annot = np.empty_like(norm_matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                count = matrix[i, j]
                percent = norm_matrix[i, j]
                annot[i, j] = f"{percent:.1%}\n({count})"

        sns.heatmap(norm_matrix, annot=annot, fmt='', cmap='Blues', 
                   xticklabels=labels, yticklabels=self.class_names, 
                   ax=ax, cbar=True, vmin=0, vmax=1, 
                   cbar_kws={'label': 'Percentage'})
        ax.set_title("Normalized Confusion Matrix (Rows sum to 100%)", fontsize=16, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=14)
        ax.set_xlabel("Predicted Label", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        safe_save_figure(fig, os.path.join(self.plots_dir, "confusion_matrix.png"))

    def _plot_score_distribution(self, predictions_file):
        with open(predictions_file, 'r') as f: 
            predictions = json.load(f)
        scores = [p['score'] for p in predictions]
        if not scores: 
            logger.warning("No prediction scores found")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(scores, ax=ax, bins=50, kde=True, alpha=0.7)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Score Threshold (0.5)')
        ax.set_xlabel("Confidence Score", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.set_title("Distribution of Prediction Scores", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        safe_save_figure(fig, os.path.join(self.plots_dir, "score_distribution.png"))

    def _generate_all_visualizations(self, predictions):
        """IMPROVED VERSION: Saves GT and Predictions in separate high-PPI files without titles."""
        logger.info(f"🖼️ Generating visualizations for all {len(self.dataset)} validation images...")
        preds_by_image = defaultdict(list)
        for p in predictions: 
            preds_by_image[p['image_id']].append(p)

        # Sort predictions by score for better visualization
        for image_id in preds_by_image:
            preds_by_image[image_id] = sorted(preds_by_image[image_id], 
                                            key=lambda x: x['score'], reverse=True)

        successful_viz = 0
        failed_viz = 0
        
        for i, input_dict in enumerate(tqdm(self.dataset, desc="Generating visualizations")):
            try:
                img_path = input_dict["file_name"]
                if not os.path.exists(img_path):
                    logger.warning(f"Image file not found: {img_path}")
                    failed_viz += 1
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load image: {img_path}")
                    failed_viz += 1
                    continue
                    
                image_id = input_dict["image_id"]
                base_filename = os.path.splitext(os.path.basename(input_dict['file_name']))[0]
                
                # -------------------------
                # 1. Ground Truth Visualization
                # -------------------------
                # High PPI separate figure, no title
                fig_gt, ax_gt = plt.subplots(figsize=(14, 10), dpi=300)
                
                v_gt = Visualizer(img[:, :, ::-1], self.metadata, scale=1.2)
                gt_img = v_gt.draw_dataset_dict(input_dict).get_image()
                ax_gt.imshow(gt_img)
                # No Title
                ax_gt.axis("off")
                
                viz_filename_gt = f"viz_{i:03d}_{base_filename}_gt.png"
                safe_save_figure(fig_gt, os.path.join(self.viz_dir, viz_filename_gt))
                
                # -------------------------
                # 2. Predictions Visualization
                # -------------------------
                # High PPI separate figure, no title
                fig_pred, ax_pred = plt.subplots(figsize=(14, 10), dpi=300)
                
                v_pred = Visualizer(img[:, :, ::-1], self.metadata, scale=1.2)
                instances = Instances(img.shape[:2])
                
                if image_id in preds_by_image:
                    preds = preds_by_image[image_id]
                    # Filter predictions with score >= 0.5 for visualization
                    high_conf_preds = [p for p in preds if p['score'] >= 0.5]
                    
                    if high_conf_preds:
                        instances.scores = torch.tensor([p['score'] for p in high_conf_preds])
                        instances.pred_classes = torch.tensor([self.coco_id_to_contiguous_id[p['category_id']] 
                                                             for p in high_conf_preds])
                        instances.pred_masks = torch.from_numpy(np.asarray([mask_util.decode(p['segmentation']) 
                                                                          for p in high_conf_preds]))
                        boxes_xywh = torch.tensor([p['bbox'] for p in high_conf_preds], dtype=torch.float32)
                        instances.pred_boxes = Boxes(BoxMode.convert(boxes_xywh, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
                        
                        pred_img = v_pred.draw_instance_predictions(instances).get_image()
                    else:
                        pred_img = img[:, :, ::-1]  # Original image if no high-confidence predictions
                else:
                    pred_img = img[:, :, ::-1]  # Original image if no predictions
                
                ax_pred.imshow(pred_img)
                # No Title
                ax_pred.axis("off")
                
                viz_filename_pred = f"viz_{i:03d}_{base_filename}_pred.png"
                safe_save_figure(fig_pred, os.path.join(self.viz_dir, viz_filename_pred))

                successful_viz += 1
                
            except Exception as e:
                logger.error(f"Failed to generate visualization for image {i}: {e}")
                failed_viz += 1
                continue
        
        logger.info(f"✅ Generated {successful_viz} pair(s) of visualizations successfully")
        if failed_viz > 0:
            logger.warning(f"⚠️ Failed to generate {failed_viz} visualizations")

    def _generate_summary_report(self, metrics, coco_results):
        """Enhanced summary report with more details"""
        report_path = os.path.join(self.output_dir, "evaluation_summary_report.md")
        
        with open(report_path, "w") as f:
            f.write(f"# Mask2Former Evaluation Report\n\n")
            f.write(f"**Dataset:** {self.dataset_name}\n")
            f.write(f"**Number of validation images:** {len(self.dataset)}\n")
            f.write(f"**Classes:** {', '.join(self.class_names)}\n\n")
            
            # COCO Metrics Summary
            f.write(f"## 📊 COCO Evaluation Metrics\n\n")
            f.write("![COCO Metrics](./plots/coco_metrics_summary.png)\n\n")
            
            if coco_results:
                f.write("### Detailed COCO Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in sorted(coco_results.items()):
                    f.write(f"| {metric} | {value:.3f} |\n")
                f.write("\n")
            
            # Class-specific performance
            f.write(f"## 📈 Class-Specific Performance\n\n")
            f.write("![Per-Class Performance](./plots/per_class_performance.png)\n\n")
            
            f.write("### Performance Metrics (IoU > 0.5, Score > 0.5)\n\n")
            f.write("| Class | Precision | Recall | F1-Score | TP | FP | FN |\n")
            f.write("|-------|-----------|--------|----------|----|----|----|\n")
            for name in self.class_names:
                m = metrics[name]
                f.write(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |\n")
            f.write("\n")
            
            # Analysis section
            f.write(f"## 🎯 Analysis\n\n")
            f.write("![Confusion Matrix](./plots/confusion_matrix.png)\n\n")
            f.write("![Score Distribution](./plots/score_distribution.png)\n\n")
            
            # Training curves
            f.write(f"## 📉 Training Progress\n\n")
            f.write("![Training Loss](./plots/training_loss_curve.png)\n\n")
            f.write("![Validation mAP](./plots/validation_map_curve.png)\n\n")
            f.write("![Learning Rate](./plots/learning_rate_curve.png)\n\n")
            
            # Key findings
            f.write("## 🔍 Key Findings\n\n")
            
            # Best performing class
            best_class = max(self.class_names, key=lambda x: metrics[x]['f1'])
            worst_class = min(self.class_names, key=lambda x: metrics[x]['f1'])
            
            f.write(f"- **Best performing class:** {best_class} (F1: {metrics[best_class]['f1']:.3f})\n")
            f.write(f"- **Worst performing class:** {worst_class} (F1: {metrics[worst_class]['f1']:.3f})\n")
            
            overall_precision = np.mean([metrics[name]['precision'] for name in self.class_names])
            overall_recall = np.mean([metrics[name]['recall'] for name in self.class_names])
            overall_f1 = np.mean([metrics[name]['f1'] for name in self.class_names])
            
            f.write(f"- **Overall macro-averaged precision:** {overall_precision:.3f}\n")
            f.write(f"- **Overall macro-averaged recall:** {overall_recall:.3f}\n")
            f.write(f"- **Overall macro-averaged F1:** {overall_f1:.3f}\n")
            
        logger.info(f"📄 Generated comprehensive summary report: {report_path}")

def setup_cfg(args):
    """Creates a config and loads settings from a file."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    cfg.defrost()
    cfg.SOLVER.BETAS = [0.9, 0.999]
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if args.model_weights: 
        cfg.MODEL.WEIGHTS = args.model_weights
        
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Mask2Former Comprehensive Evaluation Script")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-weights", required=True, help="Path to model weights")
    parser.add_argument("--output-dir", required=True, help="Base output directory for results")
    parser.add_argument("--training-log", required=True, help="Path to training log file")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options")
    args = parser.parse_args()

    logger.info("🚀 Starting Mask2Former Comprehensive Evaluation")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Model: {args.model_weights}")
    logger.info(f"Base Output: {args.output_dir}")

    datasets.register_datasets()
    base_cfg = setup_cfg(args)
    base_output_dir = args.output_dir
    
    datasets_to_evaluate = ["instance_val", "instance_test"]
    logger.info(f"Scheduled to evaluate on: {datasets_to_evaluate}")

    for dataset_name in datasets_to_evaluate:
        logger.info("="*80)
        logger.info(f"🚀 STARTING EVALUATION FOR DATASET: {dataset_name}")
        logger.info("="*80)

        # 1. Create a dataset-specific output directory
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 2. Create and configure a new CfgNode for this dataset
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.DATASETS.TEST = (dataset_name,)
        # Also update the output dir in the config, as some D2 components might use it
        cfg.OUTPUT_DIR = dataset_output_dir 
        cfg.freeze()
        
        # 3. Instantiate and run the evaluator
        evaluator = ComprehensiveEvaluator(cfg, dataset_output_dir, args.training_log)
        evaluator.run()
        
        logger.info(f"✅ Evaluation for {dataset_name} finished.")
    
    logger.info("🎉 All evaluations completed successfully!")

if __name__ == "__main__":
    main()
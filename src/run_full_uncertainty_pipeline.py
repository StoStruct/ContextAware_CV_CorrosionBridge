#!/usr/bin/env python3
"""
Integrated Uncertainty Quantification and Propagation Pipeline.
Focused on Epistemic Uncertainty (Model Uncertainty) with Publication-Ready Visualizations.
Fixed Scales: 
- Instance Uncertainty (Variance): 0.0 - 0.25
- Semantic Uncertainty (Mutual Info): 0.0 - 0.04 (Optimized for contrast)
"""

import os
import sys
import json
import logging
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import textwrap

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# --- Project Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
if os.path.exists(MASK2FORMER_ROOT):
    sys.path.insert(0, MASK2FORMER_ROOT)

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
import datasets
import datasets_semantic

# --- Publication Quality Plotting Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine available serif font
import matplotlib.font_manager
available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
if 'Times New Roman' in available_fonts:
    chosen_font = 'Times New Roman'
elif 'Liberation Serif' in available_fonts:
    chosen_font = 'Liberation Serif'
elif 'DejaVu Serif' in available_fonts:
    chosen_font = 'DejaVu Serif'
else:
    chosen_font = 'serif' # System default

logger.info(f"🎨 Using font family: {chosen_font}")

# Configure Matplotlib for Top-Tier Journal Quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_font],
    'font.size': 24,
    'axes.labelsize': 32,
    'axes.titlesize': 34,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.grid': False,       
    'axes.linewidth': 2.5,
    'lines.linewidth': 3.0,
    'legend.fontsize': 24,
    'legend.title_fontsize': 26,
    'axes.spines.top': True, 
    'axes.spines.right': True
})

class MCDropoutActivator:
    @staticmethod
    def activate_dropout(model):
        count = 0
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) or 'DropPath' in module.__class__.__name__:
                module.train()
                count += 1
        logger.info(f"✅ Activated {count} dropout/droppath layers for MC sampling.")
        return count

class IntegratedUncertaintyPipeline:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading instance segmentation model...")
        self.instance_cfg = self._setup_instance_config()
        self.instance_predictor = DefaultPredictor(self.instance_cfg)

        logger.info("Loading semantic segmentation model...")
        self.semantic_cfg = self._setup_semantic_config()
        self.semantic_model = build_model(self.semantic_cfg)
        self.semantic_model.eval()
        DetectionCheckpointer(self.semantic_model).load(self.semantic_cfg.MODEL.WEIGHTS)

        self.instance_metadata = MetadataCatalog.get(self.args.instance_dataset)
        self.element_classes = self.instance_metadata.thing_classes
        self.num_classes = len(self.element_classes)
        self.semantic_metadata = MetadataCatalog.get(self.args.semantic_dataset)
        self.corrosion_classes = self.semantic_metadata.stuff_classes
        self.corrosion_colors = self.semantic_metadata.stuff_colors

        # --- Aggregation Containers for Box Plots ---
        self.instance_uncertainty_agg = defaultdict(list) 
        self.semantic_uncertainty_agg = defaultdict(list)

    def _setup_instance_config(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.SOLVER.BETAS = [0.9, 0.999]
        cfg.merge_from_file(self.args.instance_config)
        cfg.MODEL.WEIGHTS = self.args.instance_weights
        if not hasattr(cfg.MODEL, "ROI_HEADS"):
             cfg.MODEL.ROI_HEADS = CN()
             cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.confidence_threshold
        cfg.freeze()
        return cfg

    def _setup_semantic_config(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.args.semantic_config)
        cfg.MODEL.WEIGHTS = self.args.semantic_weights
        cfg.freeze()
        return cfg

    def _get_instance_samples(self, image: np.ndarray) -> List[Dict]:
        samples = []
        for _ in range(self.args.num_mc_samples):
            with torch.no_grad():
                outputs = self.instance_predictor(image)
                samples.append({
                    'instances': outputs["instances"].to("cpu"),
                    'image_shape': image.shape[:2]
                })
        return samples

    def _get_semantic_samples(self, image: np.ndarray) -> torch.Tensor:
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        softmax_samples = []
        with torch.no_grad():
            for _ in range(self.args.num_mc_samples):
                outputs = self.semantic_model(inputs)
                logits = outputs[0]["sem_seg"]
                softmax = torch.nn.functional.softmax(logits, dim=0)
                softmax_samples.append(softmax.cpu())
        
        return torch.stack(softmax_samples)

    def _propagate_for_image(self, instance_samples: List[Dict], semantic_softmax_stack: torch.Tensor) -> Dict:
        T = len(instance_samples)
        ratio_samples = defaultdict(lambda: defaultdict(list))

        for _ in range(self.args.num_propagation_trials):
            i = random.randint(0, T - 1)
            sampled_instance_set = instance_samples[i]['instances']
            j = random.randint(0, T - 1)
            sampled_semantic_mask = semantic_softmax_stack[j].argmax(dim=0).numpy()

            pred_masks = sampled_instance_set.pred_masks.numpy()
            pred_classes = sampled_instance_set.pred_classes.numpy()
            pred_scores = sampled_instance_set.scores.numpy()

            for element_idx, (mask, class_id, score) in enumerate(zip(pred_masks, pred_classes, pred_scores)):
                if score < self.args.confidence_threshold: continue
                
                element_area = mask.sum()
                if element_area == 0: continue
                
                element_name = self.element_classes[class_id]
                for corrosion_idx, corrosion_name in enumerate(self.corrosion_classes):
                    if corrosion_name == "Background": continue
                    
                    corrosion_mask = (sampled_semantic_mask == corrosion_idx)
                    intersection = np.logical_and(mask, corrosion_mask).sum()
                    ratio = intersection / element_area
                    
                    element_key = f"{element_name}_{element_idx}"
                    ratio_samples[element_key][corrosion_name].append(ratio)
        
        return self._compute_ratio_statistics(ratio_samples)

    def _compute_ratio_statistics(self, ratio_samples: Dict) -> Dict:
        statistics = {}
        for element_key, corrosion_dict in ratio_samples.items():
            statistics[element_key] = {}
            for corrosion_name, ratios in corrosion_dict.items():
                if not ratios: continue
                ratios_arr = np.array(ratios) * 100
                mean, std = np.mean(ratios_arr), np.std(ratios_arr, ddof=1)
                ci_95 = np.percentile(ratios_arr, [2.5, 97.5])
                cv = (std / (mean + 1e-6))
                
                statistics[element_key][corrosion_name] = {
                    'mean_percent': float(mean), 'std_percent': float(std),
                    'ci_95_lower_percent': float(ci_95[0]), 'ci_95_upper_percent': float(ci_95[1]),
                    'cv_ratio': float(cv)
                }
        return statistics

    # --- VISUALIZATION HELPERS ---
    def _save_heatmap(self, data: np.ndarray, cmap: str, save_path: Path, vmin=None, vmax=None, title=None, cbar_label=None):
        """Saves a high-quality heatmap with a clear legend."""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, rotation=270, labelpad=25, fontsize=18)
        
        cbar.ax.tick_params(labelsize=16)

        if title:
            ax.set_title(title, pad=15, fontsize=20)
            
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.close(fig)

    def _save_segmentation_map(self, image_rgb: np.ndarray, save_path: Path):
        """Saves a clean segmentation/prediction map."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_rgb)
        ax.axis("off")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig)

    # --- INSTANCE UNCERTAINTY ---
    def _visualize_instance_uncertainty(self, image_id: str, image: np.ndarray, instance_samples: List[Dict]):
        H, W = image.shape[:2]
        T = len(instance_samples)
        mask_stack = np.zeros((self.num_classes, T, H, W), dtype=np.float32)

        for t, sample in enumerate(instance_samples):
            instances = sample['instances']
            for mask, cls in zip(instances.pred_masks, instances.pred_classes):
                if cls < self.num_classes:
                    mask_stack[cls, t] = np.logical_or(mask_stack[cls, t], mask.numpy())
        
        mean_masks_per_class = mask_stack.mean(axis=1)
        variance_maps_per_class = mask_stack.var(axis=1)
        global_epistemic_map = variance_maps_per_class.max(axis=0)

        # --- AGGREGATE STATS ---
        for cls_id in range(self.num_classes):
            mean_mask = mean_masks_per_class[cls_id]
            binary_region = mean_mask > 0.1
            if binary_region.sum() > 0:
                class_variance = variance_maps_per_class[cls_id][binary_region]
                mean_uncertainty = np.mean(class_variance)
                self.instance_uncertainty_agg[self.element_classes[cls_id]].append(mean_uncertainty)

        # --- VISUALIZATION ---
        try:
            # 1. Mean Prediction
            v = Visualizer(image[:, :, ::-1], self.instance_metadata, scale=1.0)
            mean_instances = self._masks_to_instances(mean_masks_per_class, W, H)
            mean_pred_vis = v.draw_instance_predictions(mean_instances).get_image()
            self._save_segmentation_map(mean_pred_vis, self.viz_dir / f"instance_{image_id}_mean.png")
            
            # 2. Epistemic Uncertainty Map
            self._save_heatmap(
                global_epistemic_map, 
                cmap='viridis', 
                save_path=self.viz_dir / f"instance_{image_id}_epistemic.png", 
                vmin=0, vmax=0.25,
                title="Instance Epistemic Uncertainty (Variance)",
                cbar_label="Variance"
            )
        except Exception as e:
            logger.error(f"Failed to save instance plots for {image_id}: {e}")

    def _masks_to_instances(self, mean_masks_per_class, W, H):
        from detectron2.structures import Instances, Boxes
        instances = Instances(image_size=(H, W))
        masks, classes = [], []
        for cls_id, mean_mask in enumerate(mean_masks_per_class):
            if mean_mask.max() > self.args.confidence_threshold:
                binary_mask = (mean_mask > self.args.confidence_threshold)
                masks.append(binary_mask)
                classes.append(cls_id)
        
        if masks:
            instances.pred_masks = torch.from_numpy(np.stack(masks))
            instances.pred_classes = torch.tensor(classes)
            instances.pred_boxes = Boxes(torch.zeros(len(masks), 4))
            instances.scores = torch.ones(len(masks))
        return instances

    # --- SEMANTIC UNCERTAINTY ---
    def _visualize_semantic_uncertainty(self, image_id: str, image_rgb: np.ndarray, softmax_stack: torch.Tensor):
        mean_softmax = softmax_stack.mean(dim=0)
        mean_prediction = mean_softmax.argmax(dim=0).numpy()
        
        predictive_entropy = -torch.sum(mean_softmax * torch.log2(mean_softmax + 1e-9), dim=0)
        entropy_per_sample = -torch.sum(softmax_stack * torch.log2(softmax_stack + 1e-9), dim=1)
        expected_entropy = entropy_per_sample.mean(dim=0)
        mutual_information = (predictive_entropy - expected_entropy).numpy()

        # --- AGGREGATE STATS ---
        for cls_idx, cls_name in enumerate(self.corrosion_classes):
            # Display "Background" as "No Corrosion"
            display_name = "No Corrosion" if cls_name == "Background" else cls_name
            
            cls_mask = (mean_prediction == cls_idx)
            if cls_mask.sum() > 0:
                class_mi_values = mutual_information[cls_mask]
                mean_uncertainty = np.mean(class_mi_values)
                self.semantic_uncertainty_agg[display_name].append(mean_uncertainty)

        # --- VISUALIZATION ---
        try:
            mean_pred_vis = np.zeros_like(image_rgb)
            for i, color in enumerate(self.corrosion_colors):
                mean_pred_vis[mean_prediction == i] = color
            self._save_segmentation_map(mean_pred_vis, self.viz_dir / f"semantic_{image_id}_mean.png")
            
            self._save_heatmap(
                mutual_information, 
                cmap='cividis', 
                save_path=self.viz_dir / f"semantic_{image_id}_epistemic.png",
                title="Semantic Epistemic Uncertainty (Mutual Information)",
                cbar_label="Mutual Information (bits)",
                vmin=0,
                vmax=0.04  
            )
        except Exception as e:
            logger.error(f"Failed to save semantic plots for {image_id}: {e}")

    # --- GLOBAL BOX PLOT GENERATION ---
    def _generate_global_box_plots(self):
        logger.info("📊 Generating global uncertainty box plots...")
        # 1. Original Plots (All classes, original colors)
        self._plot_box(
            self.instance_uncertainty_agg, 
            "Instance Segmentation Epistemic Uncertainty by Class", 
            "Structural Element Class", 
            "Mean Variance per Instance",
            self.output_dir / "boxplot_instance_uncertainty.png",
            palette="viridis",
            y_tick_interval=0.02 
        )
        self._plot_box(
            self.semantic_uncertainty_agg, 
            "Semantic Segmentation Epistemic Uncertainty by Class", 
            "Corrosion Intensity Class", 
            "Mean Mutual Information (bits)",
            self.output_dir / "boxplot_semantic_uncertainty.png",
            palette="magma",
            y_tick_interval=0.002 
        )

        # 2. NEW: Filtered Plots (Min 5 nodes, Light Colors, No Dots)
        logger.info("📊 Generating filtered uncertainty box plots (min 5 samples, no dots, light colors)...")
        self._plot_box_filtered(
            self.instance_uncertainty_agg,
            "Structural Element Class",
            "Mean Variance per Instance",
            self.output_dir / "boxplot_instance_uncertainty_filtered.png",
            palette="Pastel1" 
        )
        self._plot_box_filtered(
            self.semantic_uncertainty_agg,
            "Corrosion Intensity Class",
            "Mean Mutual Information (bits)",
            self.output_dir / "boxplot_semantic_uncertainty_filtered.png",
            palette="Pastel1" 
        )

    # Original Plotter (Retained)
    def _plot_box(self, data_dict, title, xlabel, ylabel, output_path, palette, y_tick_interval=None):
        if not data_dict:
            logger.warning(f"No data to plot for {title}")
            return

        data_list = []
        for cls_name, values in data_dict.items():
            for val in values:
                data_list.append({xlabel: cls_name, ylabel: val})
        
        if not data_list: return
        df = pd.DataFrame(data_list)

        # Prepare filenames for both versions
        path_obj = Path(output_path)
        base_name = path_obj.stem
        # 1. Version without dots
        path_no_dots = path_obj.parent / f"{base_name}_no_dots.png"
        # 2. Version with dots (keep original filename as requested for "plots with dots")
        path_with_dots = path_obj

        # Create Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 1. Base Box Plot
        sns.boxplot(
            x=xlabel, y=ylabel, data=df, 
            ax=ax, palette=palette, 
            linewidth=3.0, 
            width=0.6,
            showfliers=False
        )

        # 2. UPDATED: Axis Logic (Force 0 start, Limit Ticks, Force Top Number)
        # Force bottom to 0
        ax.set_ylim(bottom=0)
        # Use MaxNLocator to get ~4 ticks
        locator = ticker.MaxNLocator(nbins=4, prune=None) 
        ax.yaxis.set_major_locator(locator)
        
        # Wrap long X-axis labels
        xticks = ax.get_xticklabels()
        new_xticklabels = [textwrap.fill(label.get_text(), width=12) for label in xticks]
        ax.set_xticklabels(new_xticklabels, rotation=0, ha='center')

        # Titles Removed per request
        ax.set_xlabel(xlabel, labelpad=20, fontweight='normal') 
        ax.set_ylabel(ylabel, labelpad=20, fontweight='normal') 
        
        # Ensure full border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color('black')

        # 3. SAVE VERSION 1: NO DOTS
        fig.savefig(path_no_dots, bbox_inches='tight', dpi=600)
        logger.info(f"💾 Saved clean box plot (no dots) to {path_no_dots}")

        # 4. Add Stripplot (Dots)
        sns.stripplot(
            x=xlabel, y=ylabel, data=df,
            ax=ax, color='black', alpha=0.6, jitter=0.15, size=8
        )
        
        ax.set_ylim(bottom=0)

        # 5. SAVE VERSION 2: WITH DOTS
        fig.savefig(path_with_dots, bbox_inches='tight', dpi=600)
        plt.close(fig)
        logger.info(f"💾 Saved box plot (with dots) to {path_with_dots}")

    # NEW: Filtered Plotter (Min 5 nodes, Light Colors, No Dots)
    def _plot_box_filtered(self, data_dict, xlabel, ylabel, output_path, palette, y_tick_interval=None):
        # 1. Filter Data (Exclude classes with < 5 values)
        filtered_data = {k: v for k, v in data_dict.items() if len(v) >= 5}
        
        if not filtered_data:
            logger.warning(f"No classes have >= 5 samples for {output_path}. Skipping plot.")
            return

        data_list = []
        for cls_name, values in filtered_data.items():
            for val in values:
                data_list.append({xlabel: cls_name, ylabel: val})
        df = pd.DataFrame(data_list)

        # Create Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Base Box Plot (Light colors, no dots)
        sns.boxplot(
            x=xlabel, y=ylabel, data=df, 
            ax=ax, palette=palette,  # Pastel/Light palette
            linewidth=3.0, 
            width=0.6,
            showfliers=False,
            boxprops=dict(edgecolor='black'),      # Black border for the box
            medianprops=dict(color='black'),       # Black median line
            whiskerprops=dict(color='black'),      # Black whiskers
            capprops=dict(color='black')           # Black caps
        )

        # Axis Logic (Force 0 start, Limit Ticks)
        ax.set_ylim(bottom=0)
        locator = ticker.MaxNLocator(nbins=4, prune=None) 
        ax.yaxis.set_major_locator(locator)
        
        # Wrap long X-axis labels
        xticks = ax.get_xticklabels()
        new_xticklabels = [textwrap.fill(label.get_text(), width=12) for label in xticks]
        ax.set_xticklabels(new_xticklabels, rotation=0, ha='center')

        # Labels (No Title)
        ax.set_xlabel(xlabel, labelpad=20, fontweight='normal') 
        ax.set_ylabel(ylabel, labelpad=20, fontweight='normal') 
        
        # Ensure full border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color('black')

        # Save (Single version: Filtered, No Dots)
        fig.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close(fig)
        logger.info(f"💾 Saved filtered box plot to {output_path}")

    def run_full_pipeline(self):
        logger.info("\n" + "="*80)
        logger.info(f"🚀 INTEGRATED UNCERTAINTY PIPELINE (Epistemic Focus) for {self.args.instance_dataset}")
        logger.info("="*80 + "\n")

        MCDropoutActivator.activate_dropout(self.instance_predictor.model)
        MCDropoutActivator.activate_dropout(self.semantic_model)

        dataset_dicts = DatasetCatalog.get(self.args.instance_dataset)
        logger.info(f"📷 Processing {len(dataset_dicts)} images...")

        all_results = {}
        for data_dict in tqdm(dataset_dicts, desc="Processing Images"):
            image_id = Path(data_dict["file_name"]).stem
            image_bgr = read_image(data_dict["file_name"], format="BGR")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            instance_samples = self._get_instance_samples(image_bgr)
            semantic_samples = self._get_semantic_samples(image_rgb)
            
            if instance_samples and semantic_samples.numel() > 0:
                final_stats = self._propagate_for_image(instance_samples, semantic_samples)
                all_results[image_id] = final_stats
                
                self._visualize_instance_uncertainty(image_id, image_bgr, instance_samples)
                self._visualize_semantic_uncertainty(image_id, image_rgb, semantic_samples)
        
        self._generate_global_box_plots()

        results_file = self.output_dir / "propagation_results.json"
        with open(results_file, 'w') as f: json.dump(all_results, f, indent=2)
        
        report_df = self._generate_probabilistic_report(all_results)
        csv_file = self.output_dir / "probabilistic_corrosion_report.csv"
        report_df.to_csv(csv_file, index=False)
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ PIPELINE COMPLETE. Visualizations at: {self.viz_dir}")

    def _determine_action_flag(self, mean: float, std: float, cv: float, corrosion_class: str) -> str:
        if cv > 1.0 or std > 15.0: return "HUMAN REVIEW REQUIRED"
        if corrosion_class == 'Severe' and mean > 30.0 and std < 5.0: return "PRIORITY 1 INSPECTION"
        if corrosion_class == 'Severe' and mean > 20.0: return "PRIORITY 2 INSPECTION"
        if corrosion_class == 'Poor' and mean > 40.0 and std < 8.0: return "PRIORITY 2 INSPECTION"
        return "LOG FOR MONITORING"

    def _generate_probabilistic_report(self, all_results: Dict) -> pd.DataFrame:
        report_rows = []
        for image_id, results in all_results.items():
            for element_key, corrosion_data in results.items():
                for corrosion_class, stats in corrosion_data.items():
                    action_flag = self._determine_action_flag(
                        stats['mean_percent'], stats['std_percent'], stats['cv_ratio']*100, corrosion_class
                    )
                    report_rows.append({
                        'Image_ID': image_id, 'Element': element_key, 'Corrosion_Class': corrosion_class,
                        'Mean_Ratio_%': f"{stats['mean_percent']:.1f}",
                        'Uncertainty_Std_%': f"{stats['std_percent']:.1f}",
                        'CV_%': f"{stats['cv_ratio']*100:.1f}",
                        '95%_CI_%': f"[{stats['ci_95_lower_percent']:.1f}, {stats['ci_95_upper_percent']:.1f}]",
                        'Action_Flag': action_flag
                    })
        
        df = pd.DataFrame(report_rows)
        if not df.empty:
            priority_map = {'PRIORITY 1 INSPECTION': 1, 'PRIORITY 2 INSPECTION': 2, 'HUMAN REVIEW REQUIRED': 3, 'LOG FOR MONITORING': 4}
            df['_sort_key'] = df['Action_Flag'].map(priority_map)
            df = df.sort_values('_sort_key').drop('_sort_key', axis=1)
        return df

def main():
    parser = argparse.ArgumentParser(description="Integrated Uncertainty Quantification Pipeline")
    parser.add_argument("--instance-config", required=True, help="Path to instance model config")
    parser.add_argument("--instance-weights", required=True, help="Path to instance model weights")
    parser.add_argument("--semantic-config", required=True, help="Path to semantic model config")
    parser.add_argument("--semantic-weights", required=True, help="Path to semantic model weights")
    parser.add_argument("--output-dir", default="output/uncertainty_analysis", help="Base output directory")
    parser.add_argument("--num-mc-samples", type=int, default=30, help="Number of MC samples per model")
    parser.add_argument("--num-propagation-trials", type=int, default=1000, help="Number of MC propagation trials")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    splits_to_run = ["val", "test"]

    logger.info("⚙️ Registering all datasets...")
    datasets.register_datasets()
    datasets_semantic.register_semantic_datasets()
    logger.info("✅ Datasets registered.")
    
    for split in splits_to_run:
        logger.info("\n" + "="*80)
        logger.info(f"🚀 STARTING UNCERTAINTY ANALYSIS FOR SPLIT: {split}")
        logger.info("="*80)
        
        split_args = argparse.Namespace(**vars(args)) 
        split_args.instance_dataset = f"instance_{split}"
        split_args.semantic_dataset = f"corrosion_{split}"
        split_args.output_dir = os.path.join(base_output_dir, split)
        
        pipeline = IntegratedUncertaintyPipeline(split_args) 
        pipeline.run_full_pipeline()
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ALL UNCERTAINTY ANALYSES COMPLETE")
    logger.info(f"📁 All final reports saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
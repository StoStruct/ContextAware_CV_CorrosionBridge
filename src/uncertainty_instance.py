#!/usr/bin/env python3
"""
Uncertainty Quantification for Instance Segmentation Model
Implements Monte Carlo Dropout for epistemic uncertainty estimation
(Memory-Efficient Version)
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# --- ADD THIS IMPORT ---
import datasets
# --- END ADDED IMPORT ---


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCDropoutActivator:
    """Utility to activate dropout layers during inference"""
    
    @staticmethod
    def activate_dropout(model):
        """
        Activates all dropout layers in the model for MC sampling.
        This includes standard Dropout and DropPath (Stochastic Depth).
        """
        count = 0
        for module in model.modules():
            # Standard dropout
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
                count += 1
            # DropPath in Swin Transformer
            elif 'DropPath' in module.__class__.__name__:
                module.train()
                count += 1
        
        logger.info(f"✅ Activated {count} dropout/droppath layers for MC sampling")
        return count


class InstanceUncertaintyQuantifier:
    """
    Monte Carlo Dropout Uncertainty Quantification for Instance Segmentation
    
    Generates T stochastic forward passes and computes:
    - Pixel-level uncertainty maps (variance)
    - Detection stability metrics
    - Per-class uncertainty statistics
    """
    
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        dataset_name: str,
        output_dir: str = "outputs/uncertainty_instance",
        num_mc_samples: int = 30,
        confidence_threshold: float = 0.5
    ):
        self.config_file = config_file
        self.weights_file = weights_file
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.num_mc_samples = num_mc_samples
        self.confidence_threshold = confidence_threshold
        
        # --- MODIFIED: Create subdirectories for incremental saves ---
        self.samples_dir = self.output_dir / "mc_samples_per_image"
        self.uncertainty_dir = self.output_dir / "uncertainties_per_image"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        self.uncertainty_dir.mkdir(exist_ok=True)
        # --- END MODIFICATION ---

        # Load configuration and model
        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        
        # Get class names from metadata
        self.metadata = MetadataCatalog.get(dataset_name)
        self.class_names = self.metadata.thing_classes
        self.num_classes = len(self.class_names)
        
        logger.info(f"🎯 Initialized for dataset: {dataset_name}")
        logger.info(f"📊 Classes: {self.class_names}")
        logger.info(f"🔢 MC samples per image: {num_mc_samples}")
    
    def _setup_config(self):
        """Setup detectron2 configuration"""
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        # Explicitly define BETAS before merging, as it's not in the base config
        cfg.SOLVER.BETAS = [0.9, 0.999]  # Default AdamW betas

        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.weights_file
        
        # Ensure ROI_HEADS exists before setting SCORE_THRESH_TEST
        if not hasattr(cfg.MODEL, "ROI_HEADS"):
             from detectron2.config import CfgNode as CN
             cfg.MODEL.ROI_HEADS = CN()
             cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.freeze()
        return cfg

    # --- MODIFIED: This function now saves incrementally ---
    def run_mc_inference(self):
        """
        Perform Monte Carlo inference on the entire dataset, saving results
        incrementally for each image to avoid memory exhaustion.
        """
        logger.info(f"🚀 Starting Monte Carlo inference with {self.num_mc_samples} samples...")
        
        MCDropoutActivator.activate_dropout(self.predictor.model)
        
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        logger.info(f"📷 Processing {len(dataset_dicts)} images")
        
        processed_count = 0
        for data_dict in tqdm(dataset_dicts, desc="MC Sampling (Incremental Save)"):
            image_id = data_dict.get("image_id", os.path.basename(data_dict["file_name"]))
            img_path = data_dict["file_name"]
            
            # Define output path for this image's samples
            output_file = self.samples_dir / f"{image_id}.pt"
            if output_file.exists():
                continue # Skip if already processed

            if not os.path.exists(img_path):
                logger.warning(f"⚠️ Image not found: {img_path}")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"⚠️ Failed to load image: {img_path}")
                continue
            
            samples = []
            for _ in range(self.num_mc_samples):
                with torch.no_grad():
                    outputs = self.predictor(img)
                    samples.append({
                        'instances': outputs["instances"].to("cpu"),
                        'image_shape': img.shape[:2]
                    })
            
            # Save this image's samples to its own file
            torch.save(samples, output_file)
            processed_count += 1
        
        logger.info(f"✅ MC inference complete. {processed_count} new images processed.")
        # This function no longer returns a large dictionary

    # --- MODIFIED: This function now loads and processes incrementally ---
    def compute_pixel_uncertainties(self) -> Dict:
        """
        Compute pixel-level uncertainty maps by loading and processing
        per-image MC samples incrementally.
        """
        logger.info("📊 Computing pixel-level uncertainties (Incremental Load)...")
        
        sample_files = list(self.samples_dir.glob("*.pt"))
        if not sample_files:
            raise FileNotFoundError(f"No sample files found in {self.samples_dir}")

        all_uncertainties = {}
        
        for sample_file in tqdm(sample_files, desc="Computing Uncertainties"):
            image_id = sample_file.stem
            samples = torch.load(sample_file)
            
            H, W = samples[0]['image_shape']
            T = len(samples)
            
            mask_stack = np.zeros((self.num_classes, T, H, W), dtype=np.float32)
            
            for t, sample in enumerate(samples):
                instances = sample['instances']
                pred_masks = instances.pred_masks.numpy()
                pred_classes = instances.pred_classes.numpy()
                pred_scores = instances.scores.numpy()
                
                for mask, cls, score in zip(pred_masks, pred_classes, pred_scores):
                    if cls < self.num_classes and score >= self.confidence_threshold:
                        mask_stack[cls, t] = np.logical_or(mask_stack[cls, t], mask)
            
            mean_masks = mask_stack.mean(axis=1)
            variance_maps = mask_stack.var(axis=1)
            
            image_uncertainty_data = {
                'mean_masks': mean_masks,
                'variance_maps': variance_maps,
                'global_mean': mean_masks.max(axis=0),
                'global_variance': variance_maps.max(axis=0),
                'per_class_stats': self._compute_per_class_stats(mean_masks, variance_maps)
            }
            
            all_uncertainties[image_id] = image_uncertainty_data
            # Optional: Save each uncertainty result incrementally too
            # torch.save(image_uncertainty_data, self.uncertainty_dir / f"{image_id}.pt")

        uncertainty_file = self.output_dir / "instance_uncertainties.pt"
        torch.save(all_uncertainties, uncertainty_file)
        logger.info(f"💾 Saved aggregated uncertainty maps to {uncertainty_file}")
        
        return all_uncertainties

    def _compute_per_class_stats(
        self, 
        mean_masks: np.ndarray, 
        variance_maps: np.ndarray
    ) -> Dict:
        """Compute per-class uncertainty statistics"""
        stats = {}
        for cls_idx in range(self.num_classes):
            cls_name = self.class_names[cls_idx]
            mean_mask = mean_masks[cls_idx]
            variance_map = variance_maps[cls_idx]
            predicted_region = mean_mask > 0.5
            
            if predicted_region.sum() > 0:
                stats[cls_name] = {
                    'mean_confidence': float(mean_mask[predicted_region].mean()),
                    'mean_uncertainty': float(variance_map[predicted_region].mean()),
                    'max_uncertainty': float(variance_map[predicted_region].max()),
                    'area_pixels': int(predicted_region.sum())
                }
            else:
                stats[cls_name] = None
        return stats

    # --- MODIFIED: This function now loads incrementally ---
    def compute_detection_stability(self) -> Dict:
        """
        Compute detection stability metrics by loading and processing
        per-image MC samples incrementally.
        """
        logger.info("🎯 Computing detection stability metrics (Incremental Load)...")
        
        sample_files = list(self.samples_dir.glob("*.pt"))
        if not sample_files:
            raise FileNotFoundError(f"No sample files found in {self.samples_dir}")

        stability_results = {}
        
        for sample_file in tqdm(sample_files, desc="Detection Stability"):
            image_id = sample_file.stem
            samples = torch.load(sample_file)
            T = len(samples)
            
            class_detection_counts = defaultdict(list)
            for sample in samples:
                instances = sample['instances']
                pred_classes = instances.pred_classes.numpy()
                pred_scores = instances.scores.numpy()
                
                for cls in range(self.num_classes):
                    count = ((pred_classes == cls) & 
                            (pred_scores >= self.confidence_threshold)).sum()
                    class_detection_counts[cls].append(count)
            
            per_class_stability = {}
            for cls_idx in range(self.num_classes):
                counts = class_detection_counts[cls_idx]
                cls_name = self.class_names[cls_idx]
                
                per_class_stability[cls_name] = {
                    'mean_detections': float(np.mean(counts)),
                    'std_detections': float(np.std(counts)),
                    'min_detections': int(np.min(counts)),
                    'max_detections': int(np.max(counts)),
                    'detection_rate': float(np.mean([c > 0 for c in counts]))
                }
            
            stability_results[image_id] = per_class_stability
        
        stability_file = self.output_dir / "detection_stability.json"
        with open(stability_file, 'w') as f:
            json.dump(stability_results, f, indent=2)
        logger.info(f"💾 Saved stability metrics to {stability_file}")
        
        return stability_results
    
    def generate_summary_report(self, uncertainties: Dict, stability: Dict):
        # This function remains largely the same as it processes the final aggregated dictionaries
        logger.info("📋 Generating summary report...")
        all_class_uncertainties = defaultdict(list)
        all_class_stabilities = defaultdict(list)
        
        for image_id in uncertainties.keys():
            per_class_stats = uncertainties[image_id]['per_class_stats']
            per_class_stability = stability[image_id]
            
            for cls_name in self.class_names:
                if per_class_stats.get(cls_name) is not None:
                    all_class_uncertainties[cls_name].append(per_class_stats[cls_name]['mean_uncertainty'])
                if cls_name in per_class_stability:
                    all_class_stabilities[cls_name].append(per_class_stability[cls_name]['std_detections'])
        
        summary = {
            'dataset': self.dataset_name,
            'num_images': len(uncertainties),
            'num_mc_samples': self.num_mc_samples,
            'confidence_threshold': self.confidence_threshold,
            'per_class_summary': {}
        }
        
        for cls_name in self.class_names:
            if cls_name in all_class_uncertainties and all_class_uncertainties[cls_name]:
                summary['per_class_summary'][cls_name] = {
                    'mean_pixel_uncertainty': float(np.mean(all_class_uncertainties[cls_name])),
                    'median_pixel_uncertainty': float(np.median(all_class_uncertainties[cls_name])),
                    'mean_detection_std': float(np.mean(all_class_stabilities[cls_name])) if all_class_stabilities[cls_name] else 0.0,
                    'num_images_with_detections': len(all_class_uncertainties[cls_name])
                }
        
        summary_file = self.output_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Saved summary report to {summary_file}")
        logger.info("\n" + "="*80)
        logger.info("📊 UNCERTAINTY SUMMARY")
        logger.info("="*80)
        for cls_name, stats in summary['per_class_summary'].items():
            logger.info(f"\n{cls_name}:")
            logger.info(f"  Mean Pixel Uncertainty: {stats['mean_pixel_uncertainty']:.4f}")
            logger.info(f"  Detection Std: {stats['mean_detection_std']:.2f}")
            logger.info(f"  Images with detections: {stats['num_images_with_detections']}")

    # --- MODIFIED: Main analysis workflow is now sequential ---
    def run_full_analysis(self):
        """Run complete uncertainty quantification pipeline sequentially."""
        logger.info("\n" + "="*80)
        logger.info("🚀 INSTANCE SEGMENTATION UNCERTAINTY QUANTIFICATION")
        logger.info("="*80 + "\n")
        
        # Step 1: MC Inference (saves incrementally)
        self.run_mc_inference()
        
        # Step 2: Compute pixel uncertainties (loads incrementally)
        uncertainties = self.compute_pixel_uncertainties()
        
        # Step 3: Compute detection stability (loads incrementally)
        stability = self.compute_detection_stability()
        
        # Step 4: Generate report from aggregated results
        self.generate_summary_report(uncertainties, stability)
        
        # --- NEW STEP: Merge individual samples into one final file for propagation step ---
        logger.info("Merging individual MC samples into a single file for propagation...")
        all_samples = {}
        sample_files = list(self.samples_dir.glob("*.pt"))
        for sample_file in tqdm(sample_files, desc="Merging samples"):
            image_id = sample_file.stem
            all_samples[image_id] = torch.load(sample_file)
        
        final_samples_file = self.output_dir / "instance_mc_samples.pt"
        torch.save(all_samples, final_samples_file)
        logger.info(f"💾 Final merged samples saved to {final_samples_file}")
        # --- END NEW STEP ---

        logger.info("\n" + "="*80)
        logger.info("✅ UNCERTAINTY QUANTIFICATION COMPLETE")
        logger.info("="*80)
        logger.info(f"📁 All outputs saved to: {self.output_dir}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Instance Segmentation Uncertainty Quantification"
    )
    parser.add_argument("--config", required=True, help="Path to model config file")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--dataset", default="instance_val", help="Dataset name")
    parser.add_argument("--output-dir", default="outputs/uncertainty_instance", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of MC samples")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()

    logger.info("⚙️ Registering datasets...")
    datasets.register_datasets()
    logger.info("✅ Datasets registered.")
    
    quantifier = InstanceUncertaintyQuantifier(
        config_file=args.config,
        weights_file=args.weights,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        num_mc_samples=args.num_samples,
        confidence_threshold=args.confidence_threshold
    )
    
    quantifier.run_full_analysis()


if __name__ == "__main__":
    main()
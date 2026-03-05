#!/usr/bin/env python3
"""
Uncertainty Quantification for Semantic Segmentation Model
Implements Monte Carlo Dropout with uncertainty decomposition:
- Predictive Entropy (Total Uncertainty)
- Mutual Information (Epistemic Uncertainty)
- Expected Entropy (Aleatoric Uncertainty proxy)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict

import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticUncertaintyQuantifier:
    """
    Monte Carlo Dropout Uncertainty Quantification for Semantic Segmentation
    
    Computes three key uncertainty metrics:
    1. Predictive Entropy: Total uncertainty (epistemic + aleatoric)
    2. Mutual Information: Epistemic uncertainty (model uncertainty)
    3. Expected Entropy: Aleatoric uncertainty proxy (data uncertainty)
    """
    
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        dataset_name: str,
        output_dir: str = "outputs/uncertainty_semantic",
        num_mc_samples: int = 30
    ):
        self.config_file = config_file
        self.weights_file = weights_file
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.num_mc_samples = num_mc_samples
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup configuration
        self.cfg = self._setup_config()
        
        # Build and load model
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(self.weights_file)
        self.model.eval()
        
        # Get class names
        self.class_names = self.cfg.DATASETS.CLASS_NAMES if hasattr(self.cfg.DATASETS, 'CLASS_NAMES') \
                           else ['Background', 'Fair', 'Poor', 'Severe']
        self.num_classes = len(self.class_names)
        
        logger.info(f"🎯 Initialized for dataset: {dataset_name}")
        logger.info(f"📊 Classes: {self.class_names}")
        logger.info(f"🔢 MC samples per image: {num_mc_samples}")
    
    def _setup_config(self):
        """Setup detectron2 configuration"""
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.weights_file
        cfg.freeze()
        return cfg
    
    def _activate_dropout(self):
        """Activate dropout layers for MC sampling"""
        count = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
                count += 1
            elif 'DropPath' in module.__class__.__name__:
                module.train()
                count += 1
        logger.info(f"✅ Activated {count} dropout/droppath layers")
    
    def run_mc_inference(self) -> Dict:
        """
        Perform Monte Carlo inference and store raw softmax outputs.
        
        Returns:
            Dict mapping image_id to (T, C, H, W) softmax tensor
        """
        logger.info(f"🚀 Starting Monte Carlo inference with {self.num_mc_samples} samples...")
        
        # Activate dropout
        self._activate_dropout()
        
        # Build data loader
        from detectron2.data import DatasetCatalog
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        logger.info(f"📷 Processing {len(dataset_dicts)} images")
        
        mc_softmax_samples = {}
        
        for data_dict in tqdm(dataset_dicts, desc="MC Sampling"):
            file_name = data_dict["file_name"]
            image_id = data_dict.get("image_id", os.path.basename(file_name))
            
            if not os.path.exists(file_name):
                logger.warning(f"⚠️ Image not found: {file_name}")
                continue
            
            # Load image
            image = read_image(file_name, format=self.cfg.INPUT.FORMAT)
            height, width = image.shape[:2]
            
            softmax_samples = []
            
            for _ in range(self.num_mc_samples):
                with torch.no_grad():
                    # Prepare input
                    image_tensor = torch.as_tensor(
                        image.astype("float32").transpose(2, 0, 1)
                    )
                    inputs = [{
                        "image": image_tensor,
                        "height": height,
                        "width": width
                    }]
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    logits = outputs[0]["sem_seg"]  # (C, H, W)
                    
                    # Apply softmax and store
                    softmax = torch.nn.functional.softmax(logits, dim=0)
                    softmax_samples.append(softmax.cpu())
            
            # Stack into (T, C, H, W)
            mc_softmax_samples[image_id] = torch.stack(softmax_samples)
        
        logger.info(f"✅ MC inference complete for {len(mc_softmax_samples)} images")
        
        # Save raw samples
        samples_file = self.output_dir / "semantic_mc_samples.pt"
        torch.save(mc_softmax_samples, samples_file)
        logger.info(f"💾 Saved MC samples to {samples_file}")
        
        return mc_softmax_samples
    
    def compute_uncertainty_metrics(
        self,
        mc_softmax_samples: Dict
    ) -> Dict:
        """
        Compute three key uncertainty metrics from softmax samples.
        
        Metrics:
        1. Predictive Entropy (H): Total uncertainty
        2. Expected Entropy (E[H]): Aleatoric uncertainty proxy
        3. Mutual Information (I = H - E[H]): Epistemic uncertainty
        
        Args:
            mc_softmax_samples: Dict mapping image_id to (T, C, H, W) tensor
            
        Returns:
            Dict with uncertainty maps for each image
        """
        logger.info("📊 Computing uncertainty metrics...")
        
        all_uncertainties = {}
        
        for image_id, softmax_stack in tqdm(
            mc_softmax_samples.items(), 
            desc="Computing uncertainties"
        ):
            T, C, H, W = softmax_stack.shape
            
            # 1. Mean Predictive Probability
            mean_softmax = softmax_stack.mean(dim=0)  # (C, H, W)
            mean_prediction = mean_softmax.argmax(dim=0).numpy()  # (H, W)
            
            # 2. Predictive Entropy (Total Uncertainty)
            # H = -Σ p̄c log₂(p̄c)
            predictive_entropy = -torch.sum(
                mean_softmax * torch.log2(mean_softmax + 1e-9),
                dim=0
            ).numpy()  # (H, W)
            
            # 3. Expected Entropy (Aleatoric Uncertainty proxy)
            # E[H] = (1/T) Σ_t [-Σ_c p_c^(t) log₂(p_c^(t))]
            entropy_per_sample = -torch.sum(
                softmax_stack * torch.log2(softmax_stack + 1e-9),
                dim=1
            )  # (T, H, W)
            expected_entropy = entropy_per_sample.mean(dim=0).numpy()  # (H, W)
            
            # 4. Mutual Information (Epistemic Uncertainty)
            # I = H - E[H]
            mutual_information = predictive_entropy - expected_entropy  # (H, W)
            
            # Store all metrics
            all_uncertainties[image_id] = {
                'mean_prediction': mean_prediction,
                'predictive_entropy': predictive_entropy,
                'expected_entropy': expected_entropy,
                'mutual_information': mutual_information,
                'per_class_confidence': mean_softmax.numpy(),
                'statistics': self._compute_statistics(
                    mean_prediction,
                    predictive_entropy,
                    expected_entropy,
                    mutual_information
                )
            }
        
        # Save uncertainty maps
        uncertainty_file = self.output_dir / "semantic_uncertainties.pt"
        torch.save(all_uncertainties, uncertainty_file)
        logger.info(f"💾 Saved uncertainty maps to {uncertainty_file}")
        
        return all_uncertainties
    
    def _compute_statistics(
        self,
        prediction: np.ndarray,
        pred_entropy: np.ndarray,
        exp_entropy: np.ndarray,
        mutual_info: np.ndarray
    ) -> Dict:
        """Compute per-class statistics"""
        stats = {}
        
        for cls_idx, cls_name in enumerate(self.class_names):
            cls_mask = (prediction == cls_idx)
            
            if cls_mask.sum() > 0:
                stats[cls_name] = {
                    'pixel_count': int(cls_mask.sum()),
                    'percentage': float(cls_mask.sum() / cls_mask.size * 100),
                    'mean_predictive_entropy': float(pred_entropy[cls_mask].mean()),
                    'mean_aleatoric_uncertainty': float(exp_entropy[cls_mask].mean()),
                    'mean_epistemic_uncertainty': float(mutual_info[cls_mask].mean()),
                    'max_epistemic_uncertainty': float(mutual_info[cls_mask].max())
                }
            else:
                stats[cls_name] = None
        
        return stats
    
    def analyze_uncertainty_decomposition(
        self,
        uncertainties: Dict
    ):
        """
        Analyze the decomposition of uncertainty into epistemic and aleatoric components.
        Provides actionable insights for model improvement.
        """
        logger.info("🔍 Analyzing uncertainty decomposition...")
        
        # Aggregate statistics across all images
        total_pred_entropy = []
        total_epistemic = []
        total_aleatoric = []
        
        per_class_epistemic = {cls: [] for cls in self.class_names}
        per_class_aleatoric = {cls: [] for cls in self.class_names}
        
        for image_id, data in uncertainties.items():
            total_pred_entropy.append(data['predictive_entropy'].mean())
            total_epistemic.append(data['mutual_information'].mean())
            total_aleatoric.append(data['expected_entropy'].mean())
            
            stats = data['statistics']
            for cls_name in self.class_names:
                if stats.get(cls_name) is not None:
                    per_class_epistemic[cls_name].append(
                        stats[cls_name]['mean_epistemic_uncertainty']
                    )
                    per_class_aleatoric[cls_name].append(
                        stats[cls_name]['mean_aleatoric_uncertainty']
                    )
        
        # Compute analysis results
        analysis = {
            'overall_metrics': {
                'mean_predictive_entropy': float(np.mean(total_pred_entropy)),
                'mean_epistemic_uncertainty': float(np.mean(total_epistemic)),
                'mean_aleatoric_uncertainty': float(np.mean(total_aleatoric)),
                'epistemic_to_total_ratio': float(
                    np.mean(total_epistemic) / (np.mean(total_pred_entropy) + 1e-9)
                )
            },
            'per_class_analysis': {},
            'actionable_insights': []
        }
        
        # Per-class analysis
        for cls_name in self.class_names:
            if per_class_epistemic[cls_name]:
                mean_epistemic = np.mean(per_class_epistemic[cls_name])
                mean_aleatoric = np.mean(per_class_aleatoric[cls_name])
                
                analysis['per_class_analysis'][cls_name] = {
                    'mean_epistemic': float(mean_epistemic),
                    'mean_aleatoric': float(mean_aleatoric),
                    'epistemic_dominant': mean_epistemic > mean_aleatoric
                }
                
                # Generate insights
                if mean_epistemic > mean_aleatoric:
                    analysis['actionable_insights'].append({
                        'class': cls_name,
                        'issue': 'High epistemic uncertainty',
                        'recommendation': f'Add more training data for {cls_name} class. '
                                        f'Model lacks knowledge about this class.'
                    })
                else:
                    analysis['actionable_insights'].append({
                        'class': cls_name,
                        'issue': 'High aleatoric uncertainty',
                        'recommendation': f'Improve data quality for {cls_name}. '
                                        f'Issue likely due to noisy labels or inherent ambiguity.'
                    })
        
        # Save analysis
        analysis_file = self.output_dir / "uncertainty_decomposition_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"💾 Saved decomposition analysis to {analysis_file}")
        
        # Print insights
        logger.info("\n" + "="*80)
        logger.info("🔍 UNCERTAINTY DECOMPOSITION ANALYSIS")
        logger.info("="*80)
        logger.info(f"\nOverall:")
        logger.info(f"  Total Uncertainty: {analysis['overall_metrics']['mean_predictive_entropy']:.4f}")
        logger.info(f"  Epistemic (Model): {analysis['overall_metrics']['mean_epistemic_uncertainty']:.4f}")
        logger.info(f"  Aleatoric (Data):  {analysis['overall_metrics']['mean_aleatoric_uncertainty']:.4f}")
        
        logger.info("\n📋 ACTIONABLE INSIGHTS:")
        for insight in analysis['actionable_insights']:
            logger.info(f"\n  {insight['class']}:")
            logger.info(f"    Issue: {insight['issue']}")
            logger.info(f"    → {insight['recommendation']}")
    
    def run_full_analysis(self):
        """Run complete uncertainty quantification pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🚀 SEMANTIC SEGMENTATION UNCERTAINTY QUANTIFICATION")
        logger.info("="*80 + "\n")
        
        # Step 1: MC Inference
        mc_samples = self.run_mc_inference()
        
        # Step 2: Compute uncertainty metrics
        uncertainties = self.compute_uncertainty_metrics(mc_samples)
        
        # Step 3: Analyze uncertainty decomposition
        self.analyze_uncertainty_decomposition(uncertainties)
        
        logger.info("\n" + "="*80)
        logger.info("✅ UNCERTAINTY QUANTIFICATION COMPLETE")
        logger.info("="*80)
        logger.info(f"📁 All outputs saved to: {self.output_dir}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Uncertainty Quantification"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config file"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--dataset",
        default="corrosion_val",
        help="Dataset name (default: corrosion_val)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/uncertainty_semantic",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of MC samples (default: 30)"
    )
    
    args = parser.parse_args()
    
    quantifier = SemanticUncertaintyQuantifier(
        config_file=args.config,
        weights_file=args.weights,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        num_mc_samples=args.num_samples
    )
    
    quantifier.run_full_analysis()


if __name__ == "__main__":
    main()
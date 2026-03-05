#!/usr/bin/env python3
"""
Uncertainty Propagation for Corrosion Ratio Calculation

Implements Monte Carlo simulation to propagate uncertainties from both
instance and semantic segmentation models through the final ratio calculation.

Ratio = Area(Mask_instance ∩ Mask_semantic) / Area(Mask_instance)
"""

import os
import json
import logging
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UncertaintyPropagator:
    """
    Monte Carlo Uncertainty Propagation for Corrosion Area Ratio
    
    Propagates uncertainties from two independent models through
    a non-linear ratio calculation using Monte Carlo simulation.
    """
    
    def __init__(
        self,
        instance_samples_file: str,
        semantic_samples_file: str,
        output_dir: str = "outputs/uncertainty_propagation",
        n_propagation_trials: int = 1000,
        confidence_threshold: float = 0.5
    ):
        self.instance_samples_file = instance_samples_file
        self.semantic_samples_file = semantic_samples_file
        self.output_dir = Path(output_dir)
        self.n_trials = n_propagation_trials
        self.confidence_threshold = confidence_threshold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load MC samples
        logger.info("📂 Loading MC samples...")
        self.instance_mc_samples = torch.load(instance_samples_file)
        self.semantic_mc_samples = torch.load(semantic_samples_file)
        
        # Extract class names
        first_instance = next(iter(self.instance_mc_samples.values()))[0]['instances']
        # Assume class names are stored or use defaults
        self.element_classes = ['Bearing', 'Out of Plane Stiffener', 'Gusset Plate Connection']
        self.corrosion_classes = ['Background', 'Fair', 'Poor', 'Severe']
        
        logger.info(f"✅ Loaded samples for {len(self.instance_mc_samples)} images")
        logger.info(f"🔢 Propagation trials per ratio: {n_propagation_trials}")
    
    def propagate_uncertainty_for_image(
        self,
        image_id: str
    ) -> Dict:
        """
        Perform Monte Carlo simulation to propagate uncertainty for one image.
        
        Algorithm from document Section 6.2:
        1. For N trials:
           a. Randomly sample one instance prediction from T samples
           b. Randomly sample one semantic prediction from T samples
           c. Calculate ratio for each structural element
        2. Aggregate ratios to get distribution statistics
        
        Args:
            image_id: Image identifier
            
        Returns:
            Dict with ratio statistics for each element and corrosion class
        """
        instance_samples = self.instance_mc_samples[image_id]
        semantic_softmax = self.semantic_mc_samples[image_id]  # (T, C, H, W)
        
        T = len(instance_samples)
        
        # Store ratio samples: element_key -> corrosion_class -> list of ratios
        ratio_samples = defaultdict(lambda: defaultdict(list))
        
        # Monte Carlo simulation loop
        for trial in range(self.n_trials):
            # Randomly sample one instance prediction
            i = random.randint(0, T - 1)
            sampled_instance = instance_samples[i]['instances']
            
            # Randomly sample one semantic prediction
            j = random.randint(0, T - 1)
            sampled_semantic_softmax = semantic_softmax[j]  # (C, H, W)
            sampled_semantic_mask = sampled_semantic_softmax.argmax(dim=0).numpy()
            
            # Process each detected structural element
            pred_masks = sampled_instance.pred_masks.numpy()
            pred_classes = sampled_instance.pred_classes.numpy()
            pred_scores = sampled_instance.scores.numpy()
            
            for element_idx, (element_mask, element_class, score) in enumerate(
                zip(pred_masks, pred_classes, pred_scores)
            ):
                # Filter by confidence
                if score < self.confidence_threshold:
                    continue
                
                element_class_name = self.element_classes[element_class]
                element_area = element_mask.sum()
                
                if element_area == 0:
                    continue
                
                # Calculate intersection with each corrosion class
                for corrosion_idx in range(1, len(self.corrosion_classes)):  # Skip background
                    corrosion_name = self.corrosion_classes[corrosion_idx]
                    corrosion_mask = (sampled_semantic_mask == corrosion_idx)
                    
                    intersection = np.logical_and(element_mask, corrosion_mask)
                    intersection_area = intersection.sum()
                    
                    ratio = intersection_area / element_area
                    
                    # Store ratio sample
                    element_key = f"{element_class_name}_elem{element_idx}"
                    ratio_samples[element_key][corrosion_name].append(ratio)
        
        # Compute statistics from ratio distributions
        return self._compute_ratio_statistics(ratio_samples)
    
    def _compute_ratio_statistics(
        self,
        ratio_samples: Dict
    ) -> Dict:
        """
        Extract statistical metrics from ratio sample distributions.
        
        Computes:
        - Mean (expected value)
        - Standard deviation (uncertainty)
        - 95% confidence interval
        - Coefficient of variation
        """
        statistics = {}
        
        for element_key, corrosion_dict in ratio_samples.items():
            statistics[element_key] = {}
            
            for corrosion_name, ratios in corrosion_dict.items():
                if len(ratios) == 0:
                    continue
                
                ratios_array = np.array(ratios) * 100  # Convert to percentage
                
                mean_ratio = np.mean(ratios_array)
                std_ratio = np.std(ratios_array, ddof=1)
                ci_95 = np.percentile(ratios_array, [2.5, 97.5])
                
                # Coefficient of variation (relative uncertainty)
                cv = (std_ratio / (mean_ratio + 1e-6)) * 100
                
                statistics[element_key][corrosion_name] = {
                    'mean_%': float(mean_ratio),
                    'std_%': float(std_ratio),
                    'ci_95_lower_%': float(ci_95[0]),
                    'ci_95_upper_%': float(ci_95[1]),
                    'cv_%': float(cv),
                    'n_samples': len(ratios)
                }
        
        return statistics
    
    def _determine_action_flag(
        self,
        mean: float,
        std: float,
        cv: float,
        corrosion_class: str
    ) -> str:
        """
        Risk-based alerting system from document Section 7.1
        
        Flags:
        - PRIORITY 1 INSPECTION: High-confidence, high-risk finding
        - PRIORITY 2 INSPECTION: Moderate risk
        - HUMAN REVIEW REQUIRED: High uncertainty, unreliable prediction
        - LOG FOR MONITORING: Low risk
        """
        # High uncertainty threshold
        if cv > 100 or std > 15.0:
            return "HUMAN REVIEW REQUIRED"
        
        # High-confidence, high-risk findings
        if corrosion_class == 'Severe':
            if mean > 30.0 and std < 5.0:
                return "PRIORITY 1 INSPECTION"
            elif mean > 20.0:
                return "PRIORITY 2 INSPECTION"
        elif corrosion_class == 'Poor':
            if mean > 40.0 and std < 8.0:
                return "PRIORITY 2 INSPECTION"
        
        # Low risk
        return "LOG FOR MONITORING"
    
    def generate_probabilistic_report(
        self,
        all_results: Dict
    ) -> pd.DataFrame:
        """
        Generate probabilistic corrosion assessment report.
        Format from document Section 7.2
        """
        logger.info("📋 Generating probabilistic report...")
        
        report_rows = []
        
        for image_id, results in all_results.items():
            for element_key, corrosion_data in results.items():
                for corrosion_class, stats in corrosion_data.items():
                    mean = stats['mean_%']
                    std = stats['std_%']
                    cv = stats['cv_%']
                    ci = (stats['ci_95_lower_%'], stats['ci_95_upper_%'])
                    
                    # Determine action flag
                    action_flag = self._determine_action_flag(
                        mean, std, cv, corrosion_class
                    )
                    
                    report_rows.append({
                        'Image_ID': image_id,
                        'Element': element_key,
                        'Corrosion_Class': corrosion_class,
                        'Mean_Ratio_%': f"{mean:.1f}",
                        'Uncertainty_Std_%': f"{std:.1f}",
                        'CV_%': f"{cv:.1f}",
                        '95%_CI_%': f"[{ci[0]:.1f}, {ci[1]:.1f}]",
                        'Action_Flag': action_flag
                    })
        
        df = pd.DataFrame(report_rows)
        
        # Sort by priority
        priority_order = {
            'PRIORITY 1 INSPECTION': 1,
            'PRIORITY 2 INSPECTION': 2,
            'HUMAN REVIEW REQUIRED': 3,
            'LOG FOR MONITORING': 4
        }
        df['_sort_key'] = df['Action_Flag'].map(priority_order)
        df = df.sort_values('_sort_key').drop('_sort_key', axis=1)
        
        return df
    
    def run_full_propagation(self) -> Dict:
        """
        Run complete uncertainty propagation for all images.
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 UNCERTAINTY PROPAGATION THROUGH RATIO CALCULATION")
        logger.info("="*80 + "\n")
        
        # Find common images between instance and semantic
        common_images = set(self.instance_mc_samples.keys()) & \
                       set(self.semantic_mc_samples.keys())
        
        logger.info(f"📷 Processing {len(common_images)} common images")
        
        all_results = {}
        
        for image_id in tqdm(common_images, desc="Propagating uncertainty"):
            try:
                results = self.propagate_uncertainty_for_image(image_id)
                all_results[image_id] = results
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {image_id}: {e}")
                continue
        
        # Save raw results
        results_file = self.output_dir / "propagation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"💾 Saved propagation results to {results_file}")
        
        # Generate and save probabilistic report
        report_df = self.generate_probabilistic_report(all_results)
        
        # Save as CSV
        csv_file = self.output_dir / "probabilistic_corrosion_report.csv"
        report_df.to_csv(csv_file, index=False)
        logger.info(f"💾 Saved probabilistic report to {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / "probabilistic_corrosion_report.json"
        report_df.to_json(json_file, orient='records', indent=2)
        
        # Print summary statistics
        self._print_summary(report_df)
        
        logger.info("\n" + "="*80)
        logger.info("✅ UNCERTAINTY PROPAGATION COMPLETE")
        logger.info("="*80)
        logger.info(f"📁 All outputs saved to: {self.output_dir}")
        
        return all_results
    
    def _print_summary(self, report_df: pd.DataFrame):
        """Print summary statistics"""
        logger.info("\n" + "="*80)
        logger.info("📊 PROBABILISTIC ASSESSMENT SUMMARY")
        logger.info("="*80)
        
        # Count by action flag
        logger.info("\nAction Flag Distribution:")
        for flag, count in report_df['Action_Flag'].value_counts().items():
            logger.info(f"  {flag}: {count}")
        
        # High priority findings
        priority_1 = report_df[report_df['Action_Flag'] == 'PRIORITY 1 INSPECTION']
        if len(priority_1) > 0:
            logger.info(f"\n🚨 {len(priority_1)} PRIORITY 1 findings requiring immediate inspection")
        
        # High uncertainty findings
        review_required = report_df[report_df['Action_Flag'] == 'HUMAN REVIEW REQUIRED']
        if len(review_required) > 0:
            logger.info(f"\n⚠️  {len(review_required)} findings with high uncertainty requiring human review")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Uncertainty Propagation for Corrosion Ratio Calculation"
    )
    parser.add_argument(
        "--instance-samples",
        required=True,
        help="Path to instance MC samples (.pt file)"
    )
    parser.add_argument(
        "--semantic-samples",
        required=True,
        help="Path to semantic MC samples (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/uncertainty_propagation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Number of Monte Carlo propagation trials (default: 1000)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for instance predictions (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    propagator = UncertaintyPropagator(
        instance_samples_file=args.instance_samples,
        semantic_samples_file=args.semantic_samples,
        output_dir=args.output_dir,
        n_propagation_trials=args.n_trials,
        confidence_threshold=args.confidence_threshold
    )
    
    propagator.run_full_propagation()


if __name__ == "__main__":
    main()
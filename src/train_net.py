#!/usr/bin/env python3
"""
Main training script for Mask2Former models using detectron2.

This script provides an enhanced training loop with features such as:
- Advanced, config-driven optimizer construction.
- Detailed training monitoring with plots and image visualizations.
- Best model checkpointing based on validation mAP.
"""
import sys
import os
import logging
import random
import time
import itertools
import math
from typing import Dict, Any
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server-side execution
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

# --- Environment Settings for Stability ---
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_JIT"] = "0"
torch.jit._state.disable()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# --- Project Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
if os.path.exists(MASK2FORMER_ROOT):
    sys.path.insert(0, MASK2FORMER_ROOT)

# --- Detectron2 Core Imports ---
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    DefaultPredictor
)
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.deeplab import add_deeplab_config

# --- Mask2Former and Local Imports ---
import datasets
from mask2former import add_maskformer2_config, MaskFormerInstanceDatasetMapper

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Matplotlib Plotting Style ---
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12, 'figure.dpi': 100, 'savefig.dpi': 200,
    'figure.max_open_warning': 0, 'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2.5, 'axes.linewidth': 1.2, 'legend.frameon': True,
    'legend.fancybox': True, 'legend.shadow': True, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'figure.constrained_layout.use': True,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2
})


# ==============================================================================
#  Training Monitoring & Visualization
# ==============================================================================
class TrainingPlotter:
    """A helper class to generate and save plots of training progress."""
    def __init__(self, output_dir, class_names):
        self.plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.class_names = class_names
        self.history = defaultdict(list)
        logger.info(f"Training plotter initialized. Plots will be saved to: {self.plots_dir}")

    def update_training_data(self, iteration, train_loss, learning_rate):
        self.history["iteration"].append(iteration)
        self.history["train_loss"].append(train_loss)
        self.history["lr"].append(learning_rate)

    def update_eval_data(self, iteration, eval_results):
        self.history["eval_iteration"].append(iteration)
        segm_results = eval_results.get("segm", {})
        self.history["eval_map"].append(segm_results.get("AP", 0.0))
        for name in self.class_names:
            self.history[f"AP-{name}"].append(segm_results.get(f"AP-{name}", 0.0))

    def create_training_plots(self, current_iteration):
        try:
            self._plot_loss_and_map()
            self._plot_per_class_performance()
            self._plot_learning_rate()  # <-- ADDED
            self._plot_dashboard(current_iteration)
            logger.info(f"Training plots updated at iteration {current_iteration}")
        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")

    def _safe_save_figure(self, fig, filepath):
        try:
            fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
            
    def _ema(self, data, alpha=0.1):
        """Calculates the Exponential Moving Average."""
        if not data: return []
        smoothed, last_val = [], data[0]
        for val in data:
            last_val = alpha * val + (1 - alpha) * last_val
            smoothed.append(last_val)
        return smoothed

    def _plot_loss_and_map(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        raw_loss = self.history["train_loss"]
        ax1.plot(self.history["iteration"], raw_loss, 'b-', alpha=0.4, label='Training Loss (Raw)')
        if len(raw_loss) > 10:
            smoothed_loss = self._ema(raw_loss)
            ax1.plot(self.history["iteration"], smoothed_loss, 'b-', linewidth=2.5, label='Training Loss (Smoothed)')

        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Training Loss', color='b')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(self.history["eval_iteration"], self.history["eval_map"], 'g-o', label='Validation mAP')
        ax2.set_ylabel('Validation mAP', color='g')
        ax2.legend(loc='upper right')
        ax1.set_title('Training Loss and Validation mAP')
        self._safe_save_figure(fig, os.path.join(self.plots_dir, 'loss_map_progress.png'))

    def _plot_per_class_performance(self):
        if not self.history["eval_iteration"]: return
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.class_names)))
        for i, name in enumerate(self.class_names):
            ax.plot(self.history["eval_iteration"], self.history[f"AP-{name}"], 'o-', color=colors[i], label=name)
        ax.set_title('Per-Class AP Progress'); ax.legend(loc='best')
        self._safe_save_figure(fig, os.path.join(self.plots_dir, 'per_class_performance.png'))
    
    # --- START NEW METHOD ---
    def _plot_learning_rate(self):
        """Plots the learning rate schedule over iterations."""
        if not self.history["iteration"] or not self.history["lr"]:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.history["iteration"], self.history["lr"], 'r-', label='Learning Rate')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')  # Use a logarithmic scale for better visualization
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        self._safe_save_figure(fig, os.path.join(self.plots_dir, 'learning_rate_progress.png'))
    # --- END NEW METHOD ---

    def _plot_dashboard(self, current_iteration):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history["iteration"], self.history["train_loss"], 'b-', alpha=0.4)
        if len(self.history["train_loss"]) > 10:
             ax1.plot(self.history["iteration"], self._ema(self.history["train_loss"]), 'b-')
        ax1.set_title('Training Loss')
        
        ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(self.history["eval_iteration"], self.history["eval_map"], 'g-o'); ax2.set_title('Validation mAP')
        ax3 = fig.add_subplot(gs[1, 0])
        for name in self.class_names:
            if self.history[f"AP-{name}"]: ax3.plot(self.history["eval_iteration"], self.history[f"AP-{name}"], 'o-', label=name)
        ax3.set_title('Per-Class AP Progress'); ax3.legend(loc='best')
        ax4 = fig.add_subplot(gs[1, 1])
        latest_ap = [self.history[f"AP-{name}"][-1] for name in self.class_names if self.history[f"AP-{name}"]]
        if latest_ap:
            x_pos = np.arange(len(self.class_names)); ax4.bar(x_pos, latest_ap)
            ax4.set_title(f'Current Per-Class AP (Iter {current_iteration})')
            ax4.set_xticks(x_pos); ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        plt.suptitle(f'Training Dashboard - Iteration {current_iteration:,}', fontsize=16)
        self._safe_save_figure(fig, os.path.join(self.plots_dir, 'dashboard.png'))

class TrainingVisualizer:
    """A helper class to generate and save prediction visualizations on a sample of images."""
    def __init__(self, output_dir, dataset, metadata):
        from detectron2.data import detection_utils
        self.viz_dir = os.path.join(output_dir, "training_visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        self.dataset = dataset; self.metadata = metadata
        self.detection_utils = detection_utils
        self.sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
        logger.info(f"Training visualizer initialized. Visualizations will be saved to: {self.viz_dir}")

    def create_prediction_visualizations(self, predictor, current_iteration):
        for i, sample_idx in enumerate(self.sample_indices):
            self._visualize_single_prediction(predictor, sample_idx, current_iteration, i)

    def _visualize_single_prediction(self, predictor, sample_idx, current_iteration, sample_num):
        sample_data = self.dataset[sample_idx]
        img = self.detection_utils.read_image(sample_data["file_name"], format="RGB")
        outputs = predictor(img)
        v = Visualizer(img, self.metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        gt_visualizer = Visualizer(img, self.metadata, scale=1.0)
        gt_img = gt_visualizer.draw_dataset_dict(sample_data).get_image()
        ax1.imshow(gt_img); ax1.set_title(f'Ground Truth (Sample {sample_num + 1})'); ax1.axis('off')
        ax2.imshow(out.get_image()); ax2.set_title(f'Prediction (Iter {current_iteration})'); ax2.axis('off')
        fig.savefig(os.path.join(self.viz_dir, f'sample_{sample_num + 1}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


# ==============================================================================
#  Enhanced Trainer with Custom Hooks
# ==============================================================================
class EnhancedTrainer(DefaultTrainer):
    """
    An enhanced trainer that integrates custom logic for monitoring, checkpointing,
    and a custom training step compatible with Mask2Former.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.class_names = self.metadata.thing_classes
        self.plotter = TrainingPlotter(cfg.OUTPUT_DIR, self.class_names)
        val_dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        self.visualizer = TrainingVisualizer(cfg.OUTPUT_DIR, val_dataset, self.metadata)
        self._verify_model_architecture(cfg)

        self._best_map = 0.0
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)

        self._apply_custom_class_weights(cfg)

    def _apply_custom_class_weights(self, cfg):
        """
        Correctly applies per-class weights to the loss function's criterion
        after the model has been initialized.
        """
        logger.info("=" * 40)
        logger.info("Applying custom per-class weights to the loss function...")
        
        class_weights_dict = {
            'Bearing': 1.0049, 
            'Out of Plane Stiffener': 0.6193,
            'Gusset Plate Connection': 2.5633,
        }
        
        class_names = self.metadata.thing_classes
        num_classes = len(class_names)
        
        weight_tensor = torch.ones(num_classes + 1)
        for i, name in enumerate(class_names):
            if name in class_weights_dict:
                weight_tensor[i] = class_weights_dict[name]
        
        weight_tensor[num_classes] = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        
        weight_tensor = weight_tensor.to(self.model.device)
        
        if hasattr(self.model, "criterion") and hasattr(self.model.criterion, "empty_weight"):
            self.model.criterion.empty_weight = weight_tensor
            logger.info("✅ Successfully injected custom class weights into the criterion's 'empty_weight'.")
            for name, weight in zip(class_names, weight_tensor):
                logger.info(f"  - {name}: {weight:.4f}")
            logger.info(f"  - no object: {weight_tensor[-1]:.4f}")
        else:
            logger.warning("⚠️ Could not find 'model.criterion.empty_weight' to apply custom weights.")
        logger.info("=" * 40)

    def _verify_model_architecture(self, cfg):
        """Checks that the model backbone matches the expected pre-trained weights."""
        logger.info("Verifying model architecture...")
        try:
            test_model = build_model(cfg)
            if hasattr(test_model, 'backbone') and hasattr(test_model.backbone, 'patch_embed'):
                embed_dim = test_model.backbone.patch_embed.proj.out_channels
                if embed_dim == 128:
                    logger.info("-> Architecture check passed: Swin-Base confirmed.")
                elif embed_dim == 96:
                    raise ValueError("CRITICAL MISMATCH: Model is Swin-Tiny, but Swin-Base weights are expected.")
        except Exception as e:
            logger.error(f"Failed during model architecture verification: {e}")
            raise e

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        
        norm_module_types = (torch.nn.LayerNorm,)
        params = []
        memo = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad: continue
                if value in memo: continue
                memo.add(value)
                hyperparams = {**defaults}
                if "backbone" in key:
                    hyperparams["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in key or "absolute_pos_embed" in key:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV)
        elif optimizer_type == "ADAMW":
            return torch.optim.AdamW(params, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not supported.")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(
            dataset_name, tasks=("segm",), output_dir=os.path.join(cfg.OUTPUT_DIR, "inference")
        )
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False))

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.amp.autocast('cuda', enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.scaler.scale(losses).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        if math.isfinite(losses_reduced):
            get_event_storage().put_scalars(data_time=data_time, total_loss=losses_reduced, **loss_dict_reduced)

    def after_step(self):
        super().after_step()
        
        storage = get_event_storage()
        latest_values = storage.latest()
        if 'total_loss' in latest_values:
            loss = latest_values['total_loss'][0]
            lr = self.optimizer.param_groups[0]["lr"]
            self.plotter.update_training_data(self.iter, loss, lr)

        is_eval_period = (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0
        is_last_iter = self.iter == self.max_iter - 1
        
        if (is_eval_period or is_last_iter) and self.iter > 0:
            self.run_custom_monitoring()

    def run_custom_monitoring(self):
        """Runs our plotting and checkpointing logic after an evaluation."""
        logger.info(f"Running custom monitoring at iteration {self.iter + 1}...")
        try:
            storage = get_event_storage()
            latest_scalars = storage.latest()
            
            segm_results = {}
            for k, (v, _) in latest_scalars.items():
                if "segm/" in k:
                    segm_results[k.replace("segm/", "")] = v

            if not segm_results:
                logger.warning("No segmentation results found in storage for monitoring.")
                return

            eval_results = {"segm": segm_results}
            self.plotter.update_eval_data(self.iter + 1, eval_results)
            self.plotter.create_training_plots(self.iter + 1)
            
            predictor_cfg = self.cfg.clone()
            predictor_cfg.defrost()
            predictor_cfg.MODEL.WEIGHTS = self.checkpointer.get_checkpoint_file()
            predictor_cfg.freeze()

            predictor = DefaultPredictor(predictor_cfg)
            self.visualizer.create_prediction_visualizations(predictor, self.iter + 1)
            
            map_score = segm_results.get("AP", 0.0)
            logger.info(f"Monitoring complete. Current mAP: {map_score:.4f}")

            if map_score > self._best_map:
                self._best_map = map_score
                self.checkpointer.save("model_best")
                logger.info(f"New best model saved at iteration {self.iter + 1} with mAP: {self._best_map:.4f}")

        except Exception as e:
            logger.error(f"Custom monitoring failed: {e}", exc_info=True)

    def train(self):
        logger.info("Starting enhanced training loop...")
        try:
            super().train()
        except StopIteration as e:
            logger.info(f"Training stopped gracefully: {e}")
        finally:
            logger.info("Training complete.")


# ==============================================================================
#  Configuration Setup
# ==============================================================================
def setup_cfg(args):
    """
    Creates a detectron2 config object, loads settings from the specified
    YAML file, and applies command-line options.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    
    from fvcore.common.config import CfgNode as CN
    cfg.SOLVER.BETAS = [0.9, 0.999]

    if not hasattr(cfg, "TEST"): cfg.TEST = CN()
    cfg.TEST.OBJECT_MASK_THRESHOLD = 0.8
    cfg.TEST.OVERLAP_THRESHOLD = 0.8
    if not hasattr(cfg, "AUG"): cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False

    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.MODEL.WEIGHTS and not os.path.isabs(cfg.MODEL.WEIGHTS):
        cfg.MODEL.WEIGHTS = os.path.join(PROJECT_ROOT, cfg.MODEL.WEIGHTS)
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# ==============================================================================
#  Main Execution
# ==============================================================================
def main(args):
    """Main training and evaluation function."""
    datasets.register_datasets()
    cfg = setup_cfg(args)

    if args.eval_only:
        model = EnhancedTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return EnhancedTrainer.test(cfg, model)

    trainer = EnhancedTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    temp_cfg_for_printing = setup_cfg(args)

    print("\n" + "="*80)
    print(" Mask2Former Advanced Training Script")
    print("="*80)
    print(f"  - Config File: {args.config_file}")
    print(f"  - Output Dir:  {temp_cfg_for_printing.OUTPUT_DIR}")
    print("  - Features:")
    print("    - Robust Data Loading: Enabled via config")
    print("    - Advanced Optimizer:  Enabled")
    print("    - Live Monitoring:     Enabled (Plots & Visualizations)")
    print("    - Best Model Saving:   Enabled")
    print("    - Early Stopping:      REMOVED")
    print("="*80 + "\n")

    try:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred during training launch: {e}", exc_info=True)
        sys.exit(1)
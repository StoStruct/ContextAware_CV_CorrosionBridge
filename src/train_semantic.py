#!/usr/bin/env python3

import sys
import os
import logging
import time
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import SemSegEvaluator, DatasetEvaluator, DatasetEvaluators
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.events import get_event_storage, EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer

# Add Mask2Former path
MASK2FORMER_ROOT = os.path.join(PROJECT_ROOT, "Mask2Former")
sys.path.insert(0, MASK2FORMER_ROOT)

from mask2former import (
    add_maskformer2_config,
    SemanticSegmentorWithTTA,
    MaskFormerSemanticDatasetMapper,
)

# Import dataset registration
import datasets_semantic

# Logger will be configured by default_setup, so we just get it here
logger = logging.getLogger("detectron2")


class SemSegVisualizer(DatasetEvaluator):
    """Visualizes semantic segmentation predictions on the validation set."""
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        self.metadata = MetadataCatalog.get(dataset_name)
        
        self.color_mapping = {
            0: (0, 0, 0),       # Background - Black
            1: (0, 255, 0),     # Fair - Green
            2: (255, 255, 0),   # Poor - Yellow
            3: (255, 0, 0)      # Severe - Red
        }

    def reset(self):
        pass

    def process(self, inputs, outputs):
        for input_data, output_data in zip(inputs, outputs):
            img = cv2.imread(input_data["file_name"])
            if img is None:
                continue
                
            pred = output_data["sem_seg"].argmax(dim=0).cpu().numpy()
            vis_img = img.copy()
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            
            for class_id, color in self.color_mapping.items():
                bgr_color = (color[2], color[1], color[0])
                colored_mask[pred == class_id] = bgr_color
            
            alpha = 0.5
            cv2.addWeighted(colored_mask, alpha, vis_img, 1 - alpha, 0, vis_img)
            
            out_filename = os.path.join(self.output_dir, os.path.basename(input_data["file_name"]))
            cv2.imwrite(out_filename, vis_img)

    def evaluate(self):
        return {}


class Trainer(DefaultTrainer):
    """Enhanced trainer for semantic segmentation with custom hooks and class weights."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.class_names = self.metadata.stuff_classes

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)

        self._apply_semantic_class_weights()

        self.loss_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        self.history_dir = os.path.join(cfg.OUTPUT_DIR, "history_plots")
        os.makedirs(self.history_dir, exist_ok=True)
        self._best_miou = 0.0
        self._best_iter = 0

    def _apply_semantic_class_weights(self):
        """Applies custom class weights to the loss criterion."""
        logger.info("=" * 40)
        logger.info("Applying custom semantic class weights...")
        
        semantic_class_weights_dict = {
            'Background': 1.0,
            'Fair': 0.4252,
            'Poor': 1.6930,
            'Severe': 17.4556,
        }
        
        num_classes = len(self.class_names)
        weight_tensor = torch.ones(num_classes, dtype=torch.float32)
        
        for i, name in enumerate(self.class_names):
            weight = semantic_class_weights_dict.get(name, 1.0)
            weight_tensor[i] = weight
            logger.info(f"  - {name} (id={i}): {weight:.4f}")
        
        weight_tensor = weight_tensor.to(self.model.device)
        
        if hasattr(self.model, "criterion"):
            self.model.criterion.weight = weight_tensor
            logger.info("✅ Successfully applied class weights to criterion.")
        else:
            logger.warning("⚠️ Could not find 'model.criterion' to apply class weights.")
        
        logger.info("=" * 40)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        evaluators = [
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            ),
            SemSegVisualizer(dataset_name, output_folder)
        ]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
        
    def run_step(self):
        """Implements a correct training step with manual AMP handling."""
        assert self.model.training
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
        
        if not torch.isfinite(losses):
            logger.warning(f"Loss is {losses}, skipping this batch. Loss dict: {loss_dict}")
            self.optimizer.zero_grad()
            return

        self.optimizer.zero_grad()
        self.scaler.scale(losses).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        
        # Log metrics every 20 iterations
        if self.iter % 20 == 0:
            get_event_storage().put_scalars(total_loss=losses.item(), **loss_dict_reduced, data_time=data_time)

    def after_step(self):
        super().after_step()
        storage = get_event_storage()
        
        # **CRITICAL FIX**: Use storage.latest() for robust value retrieval
        latest_scalars = storage.latest()
        loss = latest_scalars.get("total_loss", (0.0,))[0]

        self.loss_history['iteration'].append(self.iter)
        self.loss_history['total_loss'].append(loss)
        self.loss_history['lr'].append(self.optimizer.param_groups[0]["lr"])

        is_eval_period = (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0
        is_last_iter = (self.iter + 1) == self.max_iter
        if (is_eval_period or is_last_iter):
            self._run_and_log_evaluation()

    def _run_and_log_evaluation(self):
        """Runs evaluation, logs metrics, and saves plots."""
        results = self.test(self.cfg, self.model)
        sem_seg_results = results.get("sem_seg", {})
        
        self.val_metrics_history['iteration'].append(self.iter)
        for key, value in sem_seg_results.items():
            self.val_metrics_history[key].append(value)

        current_miou = sem_seg_results.get("mIoU", 0.0)
        if current_miou > self._best_miou:
            self._best_miou = current_miou
            self._best_iter = self.iter
            self.checkpointer.save("model_best_miou")
            logger.info(f"✅ New best model saved! Iteration: {self.iter}, mIoU: {current_miou:.4f}")
        
        self._plot_histories()
        self._save_history()

    def _save_history(self):
        history_path = os.path.join(self.history_dir, "training_history.json")
        history_data = {**self.loss_history, **self.val_metrics_history}
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

    def _plot_histories(self):
        """Plots training loss, learning rate, and validation mIoU."""
        # Plot Loss and mIoU
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(self.loss_history['iteration'], self.loss_history['total_loss'], 'b-', label='Total Loss')
        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Loss', color='b')
        ax2 = ax1.twinx()
        ax2.plot(self.val_metrics_history['iteration'], self.val_metrics_history['mIoU'], 'g-o', label='Validation mIoU')
        ax2.set_ylabel('mIoU (%)', color='g')
        plt.title('Training Loss and Validation mIoU')
        fig.tight_layout(); plt.savefig(os.path.join(self.history_dir, "loss_vs_miou.png")); plt.close()

        # Plot Learning Rate
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss_history['iteration'], self.loss_history['lr'], 'r-')
        plt.xlabel('Iteration'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
        plt.yscale('log'); plt.savefig(os.path.join(self.history_dir, "lr_schedule.png")); plt.close()

def setup(args):
    """Setup config for training and evaluation."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    datasets_semantic.register_semantic_datasets(force=True)
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return Trainer.test(cfg, model)
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    logger.info(f"Command Line Args: {args}")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
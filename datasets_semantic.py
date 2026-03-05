#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import random
import logging
from pycocotools import mask as maskUtils

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define categories with colors for visualization
SEMANTIC_CATEGORIES = {
    "Background": {"id": 0, "color": (0, 0, 0)},  # Black for background
    "Fair": {"id": 1, "color": (0, 255, 0)},      # Green
    "Poor": {"id": 2, "color": (255, 255, 0)},    # Yellow
    "Severe": {"id": 3, "color": (255, 0, 0)}     # Red
}

class LabelMeToSemanticConverter:
    def __init__(self):
        self.category_map = {name: info["id"] for name, info in SEMANTIC_CATEGORIES.items()}
        self.categories = [
            {
                "id": info["id"],
                "name": name,
                "color": info["color"],
                "supercategory": "corrosion"
            }
            for name, info in SEMANTIC_CATEGORIES.items()
        ]
        logger.info(f"Initialized converter with categories: {list(self.category_map.keys())}")
    
    def _find_image_file(self, base_name, image_dir):
        """Find corresponding image file for a LabelMe JSON"""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for ext in extensions:
            path = os.path.join(image_dir, f"{base_name}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def _validate_annotation(self, shape):
        """Ensure annotation has valid structure"""
        if 'label' not in shape:
            raise ValueError("Annotation missing 'label' field")
        if shape['label'] not in self.category_map:
            logger.warning(f"Invalid label '{shape['label']}'. Valid options: {list(SEMANTIC_CATEGORIES.keys())}")
            return False
        return True
    
    def _create_semantic_mask(self, shapes, img_shape):
        """Create semantic segmentation mask from LabelMe annotations"""
        h, w = img_shape[:2]
        
        # Initialize empty semantic segmentation mask (0 is background)
        sem_seg = np.zeros((h, w), dtype=np.uint8)
        
        # Process each shape (corrosion area)
        valid_shapes = 0
        for shape in shapes:
            # Shape validation should already be done before calling this method
            label = shape['label']
            category_id = self.category_map[label]
            
            # Convert polygon points to mask
            points = np.array(shape['points'])
            
            # Create binary mask from polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            points = points.reshape((-1, 1, 2)).astype(np.int32)
            
            # Ensure points are within image bounds
            points[:, :, 0] = np.clip(points[:, :, 0], 0, w-1)
            points[:, :, 1] = np.clip(points[:, :, 1], 0, h-1)
            
            cv2.fillPoly(mask, [points], 1)
            
            # Fill the mask with class ID (higher priority classes overwrite lower ones)
            # Priority: Severe > Poor > Fair > Background
            # This ensures that overlapping annotations are handled properly
            if label == "Severe":
                sem_seg[mask == 1] = category_id
            elif label == "Poor":
                # Only overwrite background and fair
                sem_seg[(mask == 1) & (sem_seg <= 1)] = category_id
            elif label == "Fair":
                # Only overwrite background
                sem_seg[(mask == 1) & (sem_seg == 0)] = category_id
            
            valid_shapes += 1
            
        logger.debug(f"Created semantic mask with {valid_shapes} valid shapes")
        return sem_seg
    
    def convert_labelme_to_semantic(self, labelme_dir, image_dir, output_dir):
        """Main conversion function for semantic segmentation"""
        # Create necessary directories
        sem_seg_dir = os.path.join(output_dir, "sem_seg")
        os.makedirs(sem_seg_dir, exist_ok=True)
        
        dataset_dicts = []
        
        # Get all JSON files
        json_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]
        logger.info(f"Processing {len(json_files)} annotation files from {labelme_dir}")
        
        processed_count = 0
        skipped_count = 0
        
        # Process all JSON files
        for filename in sorted(json_files):
            base_name = os.path.splitext(filename)[0]
            img_path = self._find_image_file(base_name, image_dir)
            
            if not img_path:
                logger.warning(f"Skipping {filename}: No matching image found")
                skipped_count += 1
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Skipping {filename}: Invalid image {img_path}")
                skipped_count += 1
                continue
                
            height, width = img.shape[:2]
            
            # Process annotations from labelme file
            try:
                with open(os.path.join(labelme_dir, filename)) as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
                skipped_count += 1
                continue
                
            shapes = data.get('shapes', [])
            
            # Filter out invalid shapes
            valid_shapes = []
            for shape in shapes:
                if self._validate_annotation(shape) and 'points' in shape and len(shape.get('points', [])) >= 3:
                    valid_shapes.append(shape)
            
            if not valid_shapes:
                logger.info(f"No valid shapes found in {filename}, skipping this image")
                skipped_count += 1
                continue
            
            # Create semantic segmentation mask with valid shapes only
            sem_seg = self._create_semantic_mask(valid_shapes, img.shape)
            
            # Save semantic segmentation mask
            seg_filename = f"{base_name}.png"
            seg_path = os.path.join(sem_seg_dir, seg_filename)
            cv2.imwrite(seg_path, sem_seg)
            
            # Create detectron2-compatible record
            record = {
                "file_name": img_path,
                "height": height,
                "width": width,
                "image_id": base_name,
                "sem_seg_file_name": seg_path
            }
            dataset_dicts.append(record)
            processed_count += 1
        
        logger.info(f"Successfully processed {processed_count} images, skipped {skipped_count}")
        return dataset_dicts, sem_seg_dir

def visualize_semantic_annotations(dataset_name, output_dir="visualizations", max_samples=5):
    """Visualize dataset annotations for verification"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        metadata = MetadataCatalog.get(dataset_name)
        dataset_dicts = DatasetCatalog.get(dataset_name)
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_name}: {e}")
        return
    
    if len(dataset_dicts) == 0:
        logger.warning(f"No samples in dataset {dataset_name} to visualize")
        return
    
    # Use random samples for visualization
    samples = min(max_samples, len(dataset_dicts))
    sample_indices = random.sample(range(len(dataset_dicts)), samples)
    
    for idx, sample_idx in enumerate(sample_indices):
        try:
            d = dataset_dicts[sample_idx]
            img = cv2.imread(d["file_name"])
            
            if img is None:
                logger.warning(f"Could not load image: {d['file_name']}")
                continue
            
            # Get semantic segmentation
            if "sem_seg_file_name" in d:
                sem_seg = cv2.imread(d["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
                
                if sem_seg is None:
                    logger.warning(f"Could not load semantic segmentation: {d['sem_seg_file_name']}")
                    continue
                
                # Create visualizer
                visualizer = Visualizer(
                    img[:, :, ::-1],  # Convert BGR to RGB
                    metadata=metadata,
                    scale=0.8
                )
                
                # Draw on the image
                vis = visualizer.draw_sem_seg(sem_seg)
                
                # Save visualization
                output_path = os.path.join(output_dir, f"{dataset_name}_sample_{idx}.png")
                cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                logger.info(f"Saved visualization to {output_path}")
                
        except Exception as e:
            logger.error(f"Error visualizing sample {idx}: {e}")
            continue

def register_semantic_dataset_dict(dataset_dicts, name, metadata_dict=None):
    """Register a dataset from a list of dicts."""
    DatasetCatalog.register(name, lambda: dataset_dicts)
    
    # Register metadata
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)
    
    metadata = MetadataCatalog.get(name)
    if metadata_dict:
        for key, value in metadata_dict.items():
            setattr(metadata, key, value)
    
    logger.info(f"Registered dataset {name} with {len(dataset_dicts)} samples")
    return metadata

def register_semantic_datasets(base_path="datasets/semantic", force=False):
    """Register semantic segmentation datasets with Detectron2"""
    logger.info("=" * 60)
    logger.info("Registering semantic segmentation datasets")
    logger.info("=" * 60)
    
    converter = LabelMeToSemanticConverter()
    
    registered_datasets = []
    
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_path, split)
        dataset_name = f"corrosion_{split}"
        
        # Check if dataset is already registered and force is False
        if dataset_name in DatasetCatalog.list() and not force:
            logger.info(f"Dataset {dataset_name} already registered. Skipping registration.")
            registered_datasets.append(dataset_name)
            continue
            
        # Clean existing registration if needed
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
        
        # Create output directories
        output_dir = os.path.join(split_path, "annotations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Make sure the source directories exist
        labelme_dir = os.path.join(split_path, "labelme_annotations")
        image_dir = os.path.join(split_path, "images")
        
        if not os.path.exists(labelme_dir):
            os.makedirs(labelme_dir, exist_ok=True)
            logger.warning(f"Created empty directory: {labelme_dir}")
            
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)
            logger.warning(f"Created empty directory: {image_dir}")
            
        # Check if there are any files to process
        labelme_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')] if os.path.exists(labelme_dir) else []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(image_dir) else []
        
        if not labelme_files or not image_files:
            logger.warning(f"No annotation files ({len(labelme_files)}) or image files ({len(image_files)}) found for {split}")
            continue
            
        # Convert LabelMe annotations to semantic segmentation format
        dataset_dicts, sem_seg_dir = converter.convert_labelme_to_semantic(
            labelme_dir,
            image_dir,
            output_dir
        )
        
        if not dataset_dicts:
            logger.warning(f"No valid dataset entries created for {split}")
            continue
        
        # Set metadata properties
        class_names = ["Background", "Fair", "Poor", "Severe"]
        colors = [
            (0, 0, 0),      # Background - Black
            (0, 255, 0),    # Fair - Green  
            (255, 255, 0),  # Poor - Yellow
            (255, 0, 0)     # Severe - Red
        ]

        # Set metadata properties for semantic segmentation
        metadata_dict = {
            "stuff_classes": class_names,
            "stuff_colors": colors,
            "stuff_dataset_id_to_contiguous_id": {i: i for i in range(len(class_names))},
            "ignore_label": 255,  # Standard ignore label for semantic segmentation
            "evaluator_type": "sem_seg",
        }

        # Register dataset
        register_semantic_dataset_dict(dataset_dicts, dataset_name, metadata_dict)
        registered_datasets.append(dataset_name)
        
        # Generate visual verification
        vis_dir = os.path.join(split_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            visualize_semantic_annotations(
                dataset_name, 
                vis_dir,
                max_samples=min(5, len(dataset_dicts))
            )
        except Exception as e:
            logger.error(f"Failed to generate visualizations for {dataset_name}: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Dataset registration complete. Registered: {registered_datasets}")
    logger.info("=" * 60)
    
    return registered_datasets

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Generating semantic segmentation datasets")
    logger.info("=" * 60)
    
    registered = register_semantic_datasets()
    
    if registered:
        logger.info("Semantic segmentation dataset preparation complete.")
        logger.info(f"Registered datasets: {registered}")
    else:
        logger.warning("No datasets were registered!")
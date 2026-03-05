#!/usr/bin/env python3

import os
import json
import logging
import random
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed category mapping - these IDs MUST match your annotations
FIXED_CATEGORIES = {
    "Bearing": 1,
    "Out of Plane Stiffener": 2,
    "Gusset Plate Connection": 3
}

class EnhancedLabelMeToCOCOConverter:
    """Converter with robust error handling to skip invalid annotations."""
    
    def __init__(self):
        self.categories = []
        self.images = []
        self.annotations = []
        self.annotation_id = 1
        self.image_id = 1
        self.class_distribution = defaultdict(int)
        
        logger.info("Creating categories from FIXED_CATEGORIES...")
        for name, cat_id in sorted(FIXED_CATEGORIES.items(), key=lambda item: item[1]):
            self.categories.append({"id": cat_id, "name": name, "supercategory": "object"})
        logger.info(f"Categories created: {[cat['name'] for cat in self.categories]}")

    def convert_labelme_to_coco(self, labelme_dir: str, images_dir: str, output_json: str):
        logger.info(f"Converting LabelMe annotations from {labelme_dir}")
        json_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON annotation files")
        
        processed_images = 0
        
        for json_file in json_files:
            json_path = os.path.join(labelme_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    labelme_data = json.load(f)
                
                image_path = self._find_corresponding_image(json_file, labelme_data, images_dir)
                if not image_path:
                    continue
                
                with Image.open(image_path) as image:
                    width, height = image.size
                
                image_info = {"id": self.image_id, "file_name": os.path.basename(image_path), "width": width, "height": height}
                
                shapes = labelme_data.get('shapes', [])
                image_annotations = []
                for shape in shapes:
                    annotation_result = self._process_shape(shape, self.image_id, width, height)
                    if annotation_result["success"]:
                        image_annotations.append(annotation_result["annotation"])
                        self.class_distribution[shape["label"]] += 1
                
                if image_annotations:
                    self.images.append(image_info)
                    for ann in image_annotations:
                        ann['id'] = self.annotation_id
                        self.annotations.append(ann)
                        self.annotation_id += 1
                    processed_images += 1
                    self.image_id += 1
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        coco_data = {
            "info": {"description": "Instance Segmentation Dataset"},
            "licenses": [], "categories": self.categories,
            "images": self.images, "annotations": self.annotations
        }
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        self._generate_conversion_report(output_json, processed_images, len(self.annotations))

    def _find_corresponding_image(self, json_file: str, labelme_data: dict, images_dir: str) -> Optional[str]:
        image_filename = labelme_data.get('imagePath', os.path.splitext(json_file)[0])
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for ext in image_extensions:
            potential_path = os.path.join(images_dir, os.path.splitext(image_filename)[0] + ext)
            if os.path.exists(potential_path):
                return potential_path
        
        logger.warning(f"No corresponding image found for {json_file}")
        return None

    def _process_shape(self, shape: dict, image_id: int, width: int, height: int) -> dict:
        label = shape.get('label')
        if not label or label not in FIXED_CATEGORIES:
            return {"success": False}
        
        category_id = FIXED_CATEGORIES[label]
        
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            if len(points) < 3: return {"success": False}
            
            valid_points = [[max(0, min(p[0], width)), max(0, min(p[1], height))] for p in points]
            if len(valid_points) < 3: return {"success": False}
            
            segmentation = [coord for point in valid_points for coord in point]
            
            x_coords, y_coords = zip(*valid_points)
            x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
            bbox_width, bbox_height = x_max - x_min, y_max - y_min
            
            if bbox_width <= 1 or bbox_height <= 1: return {"success": False}
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(valid_points, dtype=np.int32)], 1)
            area = float(np.sum(mask))
            
            if area <= 10: return {"success": False}
            
            # Note: 'id' will be assigned later to ensure it's unique
            annotation = {
                "image_id": image_id, "category_id": category_id,
                "segmentation": [segmentation], "area": area,
                "bbox": [x_min, y_min, bbox_width, bbox_height], "iscrowd": 0
            }
            return {"success": True, "annotation": annotation}
        
        return {"success": False}

    def _generate_conversion_report(self, output_json: str, num_images: int, num_annotations: int):
        logger.info("="*60)
        logger.info("CONVERSION REPORT")
        logger.info("="*60)
        logger.info(f"Successfully processed: {num_images} images with valid annotations")
        logger.info(f"Total annotations created: {num_annotations}")
        logger.info(f"Saved to: {output_json}")
        logger.info("\nCLASS DISTRIBUTION:")
        for class_name, count in sorted(self.class_distribution.items()):
            logger.info(f"  {class_name}: {count} instances")

def register_datasets():
    """
    Registers the training, validation, and test datasets. This function handles
    converting LabelMe annotations to COCO format and then registers them
    with Detectron2, letting Detectron2 handle metadata creation automatically.
    """
    logger.info("=" * 60)
    logger.info("🚀 DATASET REGISTRATION")
    logger.info("=" * 60)
    
    base_path = os.path.join("datasets", "instance")

    for split in ["train", "val", "test"]:
        dataset_name = f"instance_{split}"
        split_path = os.path.join(base_path, split)
        images_dir = os.path.join(split_path, "images")
        annotations_dir = os.path.join(split_path, "labelme_annotations")
        output_json = os.path.join(split_path, "annotations.json")
        
        logger.info(f"\nProcessing {dataset_name}...")
        
        converter = EnhancedLabelMeToCOCOConverter()
        converter.convert_labelme_to_coco(annotations_dir, images_dir, output_json)
        
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
        
        register_coco_instances(
            dataset_name,
            {},
            os.path.abspath(output_json),
            os.path.abspath(images_dir)
        )
        
        # Trigger data loading to populate the metadata catalog fully.
        DatasetCatalog.get(dataset_name)
        
        logger.info(f"✅ Successfully registered {dataset_name}")
        metadata = MetadataCatalog.get(dataset_name)
        logger.info(f"  Classes discovered by Detectron2: {metadata.thing_classes}")
        logger.info(f"  ID mapping created by Detectron2: {metadata.thing_dataset_id_to_contiguous_id}")

    logger.info("\n🎉 Dataset registration completed!")

if __name__ == "__main__":
    register_datasets()
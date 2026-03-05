import os
import json
import numpy as np
import cv2
import random
import copy
import math
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations import DualTransform

class BridgeInspectionAugmentor:
    """
    Robust bridge inspection augmentation system using advanced techniques
    that improve model generalization rather than simple duplication.
    """
    
    def __init__(self, image_size=512):
        self.image_size = image_size
        
        # Define robust augmentation techniques
        self.augmentation_techniques = [
            'shift_scale_rotate',
            'perspective',
            'random_brightness_contrast',
            'clahe',
            'gaussian_blur',
            'motion_blur',
            'iso_noise',
            'chromatic_aberration',
            'mosaic'  # Special technique - combines multiple images
        ]
        
        # Initialize Albumentations transforms
        self._setup_albumentations_transforms()
        
        print(f"🎨 Robust Bridge Inspection Augmentor initialized")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Techniques: {len(self.augmentation_techniques)}")
        print(f"   Focus: Generalization over duplication")
    
    def _setup_albumentations_transforms(self):
        """Setup Albumentations transforms for consistent annotation handling."""
        
        # Correctly set up geometric transforms to handle polygons (as keypoints)
        self.shift_scale_rotate_transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            )
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.perspective_transform = A.Compose([
            A.Perspective(
                scale=(0.05, 0.15),
                keep_size=True,
                p=1.0
            )
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # Photometric & Color (These don't affect coordinates)
        self.brightness_contrast_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
        ])
        
        self.clahe_transform = A.Compose([
            A.CLAHE(clip_limit=4.0, p=1.0)
        ])
        
        # Artifact & Degradation
        self.gaussian_blur_transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0)
        ])
        
        self.motion_blur_transform = A.Compose([
            A.MotionBlur(blur_limit=7, p=1.0)
        ])
        
        self.iso_noise_transform = A.Compose([
            A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=1.0)
        ])
    
    def augment_image_with_technique(self, image, annotations, technique_name):
        """Apply a single robust augmentation technique to an image and its annotations"""
        
        if technique_name == 'shift_scale_rotate':
            return self._apply_geometric_transform(image, annotations, self.shift_scale_rotate_transform)
        elif technique_name == 'perspective':
            return self._apply_geometric_transform(image, annotations, self.perspective_transform)
        elif technique_name == 'random_brightness_contrast':
            return self.brightness_contrast_transform(image=image)['image'], annotations
        elif technique_name == 'clahe':
            return self.clahe_transform(image=image)['image'], annotations
        elif technique_name == 'gaussian_blur':
            return self.gaussian_blur_transform(image=image)['image'], annotations
        elif technique_name == 'motion_blur':
            return self.motion_blur_transform(image=image)['image'], annotations
        elif technique_name == 'iso_noise':
            return self.iso_noise_transform(image=image)['image'], annotations
        elif technique_name == 'chromatic_aberration':
            return self._chromatic_aberration(image, annotations)
        elif technique_name == 'mosaic':
            return image, annotations  
        else:
            return image, annotations

    def _apply_geometric_transform(self, image, annotations, transform):
        """
        Correctly applies a geometric transformation to an image and its polygon annotations.
        """
        try:
            if not annotations.get('shapes'):
                return transform(image=image)['image'], annotations

            all_keypoints, polygon_lengths, original_shapes_data = [], [], []
            for shape in annotations['shapes']:
                points = shape.get('points', [])
                if len(points) >= 3:
                    all_keypoints.extend(points)
                    polygon_lengths.append(len(points))
                    original_shapes_data.append({
                        'label': shape.get('label', 'unknown'),
                        'shape_type': shape.get('shape_type', 'polygon')
                    })

            if not all_keypoints:
                return transform(image=image)['image'], annotations
                
            transformed = transform(image=image, keypoints=all_keypoints)
            transformed_image, transformed_keypoints = transformed['image'], transformed['keypoints']

            new_annotations = copy.deepcopy(annotations)
            new_annotations['shapes'] = []
            
            current_kp_idx = 0
            for i, length in enumerate(polygon_lengths):
                polygon_points = transformed_keypoints[current_kp_idx : current_kp_idx + length]
                current_kp_idx += length
                new_shape = {
                    'label': original_shapes_data[i]['label'],
                    'points': polygon_points,
                    'shape_type': original_shapes_data[i]['shape_type']
                }
                new_annotations['shapes'].append(new_shape)

            return transformed_image, new_annotations
        except Exception as e:
            print(f"⚠️ Error applying geometric transform: {e}")
            return image, annotations
    
    def _chromatic_aberration(self, image, annotations):
        """Apply chromatic aberration to simulate lens artifacts with a visible effect."""
        try:
            h, w = image.shape[:2]
            shift = random.randint(4, 8)
            shift_r = random.randint(-shift, shift)
            shift_b = random.randint(-shift, shift)
            
            M_r = np.float32([[1, 0, shift_r], [0, 1, 0]])
            M_b = np.float32([[1, 0, shift_b], [0, 1, 0]])
            
            b, g, r = cv2.split(image)
            r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
            b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            aberrated_image = cv2.merge([b_shifted, g, r_shifted])
            return aberrated_image, annotations
        except Exception as e:
            print(f"⚠️ Error in chromatic_aberration: {e}")
            return image, annotations
    
    def create_mosaic(self, images_and_annotations_list):
        """Create a mosaic by combining 4 different training images."""
        if len(images_and_annotations_list) < 4:
            return None, None
        
        try:
            selected = random.sample(images_and_annotations_list, 4)
            half_size = self.image_size // 2
            mosaic_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            combined_annotations = {'shapes': []}
            
            offsets = [(0, 0), (half_size, 0), (0, half_size), (half_size, half_size)]

            for i, (img, ann) in enumerate(selected):
                resized_img = cv2.resize(img, (half_size, half_size))
                offset_x, offset_y = offsets[i]

                if i == 0: mosaic_image[0:half_size, 0:half_size] = resized_img
                elif i == 1: mosaic_image[0:half_size, half_size:self.image_size] = resized_img
                elif i == 2: mosaic_image[half_size:self.image_size, 0:half_size] = resized_img
                else: mosaic_image[half_size:self.image_size, half_size:self.image_size] = resized_img

                for shape in ann.get('shapes', []):
                    if 'points' in shape:
                        new_points = [[(p[0] * half_size / img.shape[1]) + offset_x, (p[1] * half_size / img.shape[0]) + offset_y] for p in shape['points']]
                        new_shape = copy.deepcopy(shape)
                        new_shape['points'] = new_points
                        combined_annotations['shapes'].append(new_shape)
            
            return mosaic_image, combined_annotations
        except Exception as e:
            print(f"⚠️ Error creating mosaic: {e}")
            return None, None


class DatasetAnalyzerAndSplitter:
    def __init__(self, images_dir, annotations_dir, output_dir='analyzed_dataset'):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        
        self.train_images_dir = os.path.join(output_dir, 'train', 'images')
        self.train_annotations_dir = os.path.join(output_dir, 'train', 'labelme_annotations')
        self.val_images_dir = os.path.join(output_dir, 'val', 'images')
        self.val_annotations_dir = os.path.join(output_dir, 'val', 'labelme_annotations')
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        
        for dir_path in [self.train_images_dir, self.train_annotations_dir, self.val_images_dir, self.val_annotations_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.dataset = []
        self.overall_distribution = {}
        self.original_train_data = []
        self.original_val_data = []
        self.augmentor = BridgeInspectionAugmentor()
        self.final_class_distribution = {}

    def load_and_analyze_dataset(self):
        print("🔍 Loading and analyzing initial dataset...")
        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in tqdm(image_files, desc="Analyzing dataset"):
            annotation_path = os.path.join(self.annotations_dir, os.path.splitext(image_file)[0] + '.json')
            if not os.path.exists(annotation_path): continue
            
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            image_class_counts = defaultdict(int)
            for shape in annotation.get('shapes', []):
                class_name = shape.get('label', 'unknown')
                if class_name != 'unknown':
                    image_class_counts[class_name] += 1
            
            if not image_class_counts: continue
            
            for class_name, count in image_class_counts.items():
                self.overall_distribution[class_name] = self.overall_distribution.get(class_name, 0) + count
                
            self.dataset.append({
                'filename': os.path.splitext(image_file)[0],
                'image_path': os.path.join(self.images_dir, image_file),
                'annotation_path': annotation_path,
                'dominant_class': max(image_class_counts, key=image_class_counts.get),
            })
        
        print("\n📊 Initial Class Distribution:")
        if not self.overall_distribution:
            print("  No classes found in annotations.")
            return
        for class_name, count in sorted(self.overall_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} instances")

    def perform_strategic_split(self, val_split=0.2, random_state=42):
        print(f"\n📂 Performing strategic stratified split...")
        random.seed(random_state)
        
        images_by_dominant_class = defaultdict(list)
        for img_data in self.dataset:
            images_by_dominant_class[img_data['dominant_class']].append(img_data)
        
        train_data, val_data = [], []
        for class_name, class_images in images_by_dominant_class.items():
            random.shuffle(class_images)
            n_val = max(1, int(len(class_images) * val_split)) if len(class_images) > 1 else 0
            val_data.extend(class_images[:n_val])
            train_data.extend(class_images[n_val:])
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        self.original_train_data = train_data
        self.original_val_data = val_data
        print(f"  Training: {len(self.original_train_data)} images | Validation: {len(self.original_val_data)} images")
        
        self._save_split_data(self.original_train_data, self.train_images_dir, self.train_annotations_dir, "Saving train split")
        self._save_split_data(self.original_val_data, self.val_images_dir, self.val_annotations_dir, "Saving val split")

    def _save_split_data(self, data, images_dir, annotations_dir, desc):
        for img_data in tqdm(data, desc=desc):
            self._copy_image_and_annotation(img_data, images_dir, annotations_dir)

    def _copy_image_and_annotation(self, img_data, images_dir, annotations_dir):
        filename = img_data['filename']
        src_image_path = img_data['image_path']
        image_ext = os.path.splitext(src_image_path)[1]
        dst_image_path = os.path.join(images_dir, f"{filename}{image_ext}")
        cv2.imwrite(dst_image_path, cv2.imread(src_image_path))
        
        with open(img_data['annotation_path'], 'r') as f:
            annotation = json.load(f)
        annotation['imagePath'] = f"../images/{filename}{image_ext}"
        if 'imageData' in annotation: del annotation['imageData']
        
        with open(os.path.join(annotations_dir, f"{filename}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)

    def apply_robust_augmentations(self):
        print(f"\n🎨 Applying augmentations to {len(self.original_train_data)} training images...")
        loaded_data = []
        for img_data in self.original_train_data:
            try:
                image = cv2.imread(os.path.join(self.train_images_dir, os.path.basename(img_data['image_path'])))
                annotation_path = os.path.join(self.train_annotations_dir, f"{img_data['filename']}.json")
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                loaded_data.append((image, annotation, img_data['filename']))
            except Exception as e:
                print(f"Could not load image/annotation for {img_data['filename']}: {e}")
                continue
        
        regular_techniques = [t for t in self.augmentor.augmentation_techniques if t != 'mosaic']
        for image, original_annotation, filename in tqdm(loaded_data, desc="Applying augmentations"):
            for technique in regular_techniques:
                aug_image, aug_annotation = self.augmentor.augment_image_with_technique(image.copy(), copy.deepcopy(original_annotation), technique)
                self._save_augmented_sample(filename, aug_image, aug_annotation, technique)
        
        if len(loaded_data) >= 4:
            num_mosaics = min(len(loaded_data) // 2, 50)
            mosaic_source_data = [(img, ann) for img, ann, _ in loaded_data]
            for i in tqdm(range(num_mosaics), desc="Creating mosaics"):
                mosaic_image, mosaic_annotation = self.augmentor.create_mosaic(mosaic_source_data)
                if mosaic_image is not None:
                    self._save_augmented_sample(f"mosaic_{i:03d}", mosaic_image, mosaic_annotation, "mosaic")

    def _save_augmented_sample(self, base_filename, image, annotation, technique):
        aug_filename = base_filename if technique == "mosaic" else f"{base_filename}_aug_{technique}"
        cv2.imwrite(os.path.join(self.train_images_dir, f"{aug_filename}.jpg"), image)
        
        annotation['imagePath'] = f"../images/{aug_filename}.jpg"
        if 'imageData' in annotation: del annotation['imageData']
        
        with open(os.path.join(self.train_annotations_dir, f"{aug_filename}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)

    def visualize_augmented_samples(self, num_samples=50):
        print(f"\n🖼️ Generating visualizations for {num_samples} augmented samples...")
        vis_dir = os.path.join(self.analysis_dir, 'augmented_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        all_aug_jsons = [f for f in os.listdir(self.train_annotations_dir) if '_aug_' in f or 'mosaic_' in f]
        if not all_aug_jsons:
            print("⚠️ No augmented files found to visualize.")
            return
        
        selected_files = random.sample(all_aug_jsons, min(num_samples, len(all_aug_jsons)))
        
        all_class_names = list(self.overall_distribution.keys())
        colors = plt.colormaps.get('hsv', len(all_class_names) + 1)
        class_colors = {name: tuple(int(c * 255) for c in colors(i)[:3]) for i, name in enumerate(all_class_names)}

        for json_file in tqdm(selected_files, desc="Creating visualizations"):
            try:
                with open(os.path.join(self.train_annotations_dir, json_file), 'r') as f:
                    annotation = json.load(f)
                image_path = os.path.join(self.train_images_dir, os.path.basename(annotation['imagePath']))
                image = cv2.imread(image_path)
                if image is None: continue

                for shape in annotation.get('shapes', []):
                    label = shape.get('label', 'unknown')
                    contour = np.array(shape['points'], dtype=np.int32).reshape((-1, 1, 2))
                    color = class_colors.get(label, (255, 255, 255))
                    cv2.polylines(image, [contour], isClosed=True, color=color, thickness=2)
                    cv2.putText(image, label, (contour[0][0][0], contour[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imwrite(os.path.join(vis_dir, os.path.splitext(json_file)[0] + '.jpg'), image)
            except Exception as e:
                print(f"❌ Error visualizing {json_file}: {e}")
        
        print(f"✅ Visualizations saved to: {vis_dir}")

    def analyze_final_dataset(self):
        print(f"\n📊 Analyzing final augmented training dataset...")
        final_class_counts = defaultdict(int)
        train_image_files = [f for f in os.listdir(self.train_images_dir) if f.lower().endswith('.jpg')]
        
        for image_file in tqdm(train_image_files, desc="Analyzing final dataset"):
            annotation_path = os.path.join(self.train_annotations_dir, os.path.splitext(image_file)[0] + '.json')
            if not os.path.exists(annotation_path): continue
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            for shape in annotation.get('shapes', []):
                label = shape.get('label', 'unknown')
                if label != 'unknown':
                    final_class_counts[label] += 1
        
        self.final_class_distribution = dict(final_class_counts)
        total_instances = sum(self.final_class_distribution.values())
        if not total_instances:
            print("  No instances found in the final dataset.")
            return

        min_count = min(self.final_class_distribution.values()) if self.final_class_distribution else 0
        max_count = max(self.final_class_distribution.values()) if self.final_class_distribution else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print("\n📈 Final Augmented Dataset Analysis:")
        print(f"  Total training images: {len(train_image_files)}")
        print(f"  Total instances: {total_instances}")
        print(f"  Final Imbalance Ratio: {imbalance_ratio:.2f}:1")

        print("\n📊 Final Class Distribution:")
        for class_name, count in sorted(self.final_class_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} instances ({(count/total_instances)*100:.1f}%)")

        self._generate_final_weight_report()

    def _generate_final_weight_report(self):
        total_instances = sum(self.final_class_distribution.values())
        n_classes = len(self.final_class_distribution)
        if total_instances == 0 or n_classes == 0: return
        
        recommended_weights = {cn: total_instances / (n_classes * c) if c > 0 else 0 for cn, c in self.final_class_distribution.items()}
        
        print(f"\n🎯 FINAL CLASS WEIGHT RECOMMENDATIONS:")
        for class_name, weight in sorted(recommended_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name:<25} → {weight:.4f}")
        
        print("\n💻 PYTORCH EXAMPLE:")
        print("```python")
        print("import torch")
        print("import torch.nn as nn")
        print("# Assuming class order is sorted alphabetically for the tensor")
        print("weight_tensor = torch.tensor([")
        for class_name in sorted(recommended_weights.keys()):
            print(f"    {recommended_weights[class_name]:.4f},  # {class_name}")
        print("], dtype=torch.float32)")
        print("criterion = nn.CrossEntropyLoss(weight=weight_tensor)")
        print("```")

    def process_dataset(self, val_split=0.2, random_state=42, apply_augmentations=True):
        print("🚀 Starting Instance Dataset Processing Pipeline...")
        self.load_and_analyze_dataset()
        if not self.dataset:
            print("❌ No valid instance data found. Exiting.")
            return False, None, None
        
        self.perform_strategic_split(val_split, random_state)
        
        if apply_augmentations:
            self.apply_robust_augmentations()
            self.visualize_augmented_samples()
            self.analyze_final_dataset()
        
        print("\n🎉 INSTANCE DATASET PROCESSING COMPLETE!")
        train_filenames = [d['filename'] for d in self.original_train_data]
        val_filenames = [d['filename'] for d in self.original_val_data]
        return True, train_filenames, val_filenames


# ============================================================================
# Restored SemanticDatasetProcessor as a full, separate class
# ============================================================================
class SemanticDatasetProcessor:
    def __init__(self, semantic_images_dir, semantic_annotations_dir, semantic_output_dir, 
                 instance_train_filenames, instance_val_filenames):
        self.images_dir = semantic_images_dir
        self.annotations_dir = semantic_annotations_dir
        self.output_dir = semantic_output_dir
        
        self.train_images_dir = os.path.join(self.output_dir, 'train', 'images')
        self.train_annotations_dir = os.path.join(self.output_dir, 'train', 'labelme_annotations')
        self.val_images_dir = os.path.join(self.output_dir, 'val', 'images')
        self.val_annotations_dir = os.path.join(self.output_dir, 'val', 'labelme_annotations')
        self.analysis_dir = os.path.join(self.output_dir, 'analysis')
        
        for dir_path in [self.train_images_dir, self.train_annotations_dir, self.val_images_dir, self.val_annotations_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.instance_train_filenames = set(instance_train_filenames)
        self.instance_val_filenames = set(instance_val_filenames)
        
        self.dataset = []
        self.overall_distribution = {}
        self.original_train_data = []
        self.original_val_data = []
        self.augmentor = BridgeInspectionAugmentor()
        self.final_class_distribution = {}
        print("🎯 Semantic Dataset Processor initialized")

    def load_and_analyze_dataset(self):
        print("🔍 Loading and analyzing initial semantic dataset...")
        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in tqdm(image_files, desc="Analyzing semantic dataset"):
            annotation_path = os.path.join(self.annotations_dir, os.path.splitext(image_file)[0] + '.json')
            if not os.path.exists(annotation_path): continue
            
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            image_class_counts = defaultdict(int)
            for shape in annotation.get('shapes', []):
                class_name = shape.get('label', 'unknown')
                if class_name != 'unknown':
                    image_class_counts[class_name] += 1
            
            if not image_class_counts: continue
            
            for class_name, count in image_class_counts.items():
                self.overall_distribution[class_name] = self.overall_distribution.get(class_name, 0) + count
                
            self.dataset.append({
                'filename': os.path.splitext(image_file)[0],
                'image_path': os.path.join(self.images_dir, image_file),
                'annotation_path': annotation_path,
            })
        
        print("\n📊 Initial Semantic Class Distribution:")
        if not self.overall_distribution:
            print("  No classes found in semantic annotations.")
            return
        for class_name, count in sorted(self.overall_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} instances")

    def apply_instance_split_to_semantic(self):
        print("\n🔄 Applying instance split to semantic dataset...")
        for img_data in self.dataset:
            filename = img_data['filename']
            if filename in self.instance_train_filenames:
                self.original_train_data.append(img_data)
            elif filename in self.instance_val_filenames:
                self.original_val_data.append(img_data)

        print(f"  Training: {len(self.original_train_data)} images | Validation: {len(self.original_val_data)} images")
        
        self._save_split_data(self.original_train_data, self.train_images_dir, self.train_annotations_dir, "Saving semantic train split")
        self._save_split_data(self.original_val_data, self.val_images_dir, self.val_annotations_dir, "Saving semantic val split")

    def _save_split_data(self, data, images_dir, annotations_dir, desc):
        for img_data in tqdm(data, desc=desc):
            self._copy_image_and_annotation(img_data, images_dir, annotations_dir)

    def _copy_image_and_annotation(self, img_data, images_dir, annotations_dir):
        filename = img_data['filename']
        src_image_path = img_data['image_path']
        image_ext = os.path.splitext(src_image_path)[1]
        dst_image_path = os.path.join(images_dir, f"{filename}{image_ext}")
        cv2.imwrite(dst_image_path, cv2.imread(src_image_path))
        
        with open(img_data['annotation_path'], 'r') as f:
            annotation = json.load(f)
        annotation['imagePath'] = f"../images/{filename}{image_ext}"
        if 'imageData' in annotation: del annotation['imageData']
        
        with open(os.path.join(annotations_dir, f"{filename}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)

    def apply_robust_augmentations(self):
        print(f"\n🎨 Applying augmentations to {len(self.original_train_data)} semantic training images...")
        loaded_data = []
        for img_data in self.original_train_data:
            try:
                image = cv2.imread(os.path.join(self.train_images_dir, os.path.basename(img_data['image_path'])))
                annotation_path = os.path.join(self.train_annotations_dir, f"{img_data['filename']}.json")
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                loaded_data.append((image, annotation, img_data['filename']))
            except Exception as e:
                print(f"Could not load semantic image/annotation for {img_data['filename']}: {e}")
                continue
        
        regular_techniques = [t for t in self.augmentor.augmentation_techniques if t != 'mosaic']
        for image, original_annotation, filename in tqdm(loaded_data, desc="Applying semantic augmentations"):
            for technique in regular_techniques:
                aug_image, aug_annotation = self.augmentor.augment_image_with_technique(image.copy(), copy.deepcopy(original_annotation), technique)
                self._save_augmented_sample(filename, aug_image, aug_annotation, technique)
        
        if len(loaded_data) >= 4:
            num_mosaics = min(len(loaded_data) // 2, 50)
            mosaic_source_data = [(img, ann) for img, ann, _ in loaded_data]
            for i in tqdm(range(num_mosaics), desc="Creating semantic mosaics"):
                mosaic_image, mosaic_annotation = self.augmentor.create_mosaic(mosaic_source_data)
                if mosaic_image is not None:
                    self._save_augmented_sample(f"semantic_mosaic_{i:03d}", mosaic_image, mosaic_annotation, "mosaic")

    def _save_augmented_sample(self, base_filename, image, annotation, technique):
        aug_filename = base_filename if technique == "mosaic" else f"{base_filename}_sem_aug_{technique}"
        cv2.imwrite(os.path.join(self.train_images_dir, f"{aug_filename}.jpg"), image)
        
        annotation['imagePath'] = f"../images/{aug_filename}.jpg"
        if 'imageData' in annotation: del annotation['imageData']
        
        with open(os.path.join(self.train_annotations_dir, f"{aug_filename}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)

    def visualize_augmented_samples(self, num_samples=50):
        print(f"\n🖼️ Generating visualizations for {num_samples} semantic augmented samples...")
        vis_dir = os.path.join(self.analysis_dir, 'augmented_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        all_aug_jsons = [f for f in os.listdir(self.train_annotations_dir) if '_sem_aug_' in f or 'semantic_mosaic_' in f]
        if not all_aug_jsons:
            print("⚠️ No semantic augmented files found to visualize.")
            return
        
        selected_files = random.sample(all_aug_jsons, min(num_samples, len(all_aug_jsons)))
        
        all_class_names = list(self.overall_distribution.keys())
        colors = plt.colormaps.get('hsv', len(all_class_names) + 1)
        class_colors = {name: tuple(int(c * 255) for c in colors(i)[:3]) for i, name in enumerate(all_class_names)}

        for json_file in tqdm(selected_files, desc="Creating semantic visualizations"):
            try:
                with open(os.path.join(self.train_annotations_dir, json_file), 'r') as f:
                    annotation = json.load(f)
                image_path = os.path.join(self.train_images_dir, os.path.basename(annotation['imagePath']))
                image = cv2.imread(image_path)
                if image is None: continue

                for shape in annotation.get('shapes', []):
                    label = shape.get('label', 'unknown')
                    contour = np.array(shape['points'], dtype=np.int32).reshape((-1, 1, 2))
                    color = class_colors.get(label, (255, 255, 255))
                    cv2.polylines(image, [contour], isClosed=True, color=color, thickness=2)
                    cv2.putText(image, label, (contour[0][0][0], contour[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imwrite(os.path.join(vis_dir, os.path.splitext(json_file)[0] + '.jpg'), image)
            except Exception as e:
                print(f"❌ Error visualizing semantic file {json_file}: {e}")
        
        print(f"✅ Semantic visualizations saved to: {vis_dir}")

    def analyze_final_dataset(self):
        print(f"\n📊 Analyzing final augmented semantic dataset...")
        final_class_counts = defaultdict(int)
        train_image_files = [f for f in os.listdir(self.train_images_dir) if f.lower().endswith('.jpg')]
        
        for image_file in tqdm(train_image_files, desc="Analyzing final semantic dataset"):
            annotation_path = os.path.join(self.train_annotations_dir, os.path.splitext(image_file)[0] + '.json')
            if not os.path.exists(annotation_path): continue
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            for shape in annotation.get('shapes', []):
                label = shape.get('label', 'unknown')
                if label != 'unknown':
                    final_class_counts[label] += 1
        
        self.final_class_distribution = dict(final_class_counts)
        total_instances = sum(self.final_class_distribution.values())
        if not total_instances:
            print("  No instances found in the final semantic dataset.")
            return

        min_count = min(self.final_class_distribution.values()) if self.final_class_distribution else 0
        max_count = max(self.final_class_distribution.values()) if self.final_class_distribution else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print("\n📈 Final Augmented Semantic Dataset Analysis:")
        print(f"  Total training images: {len(train_image_files)}")
        print(f"  Total instances: {total_instances}")
        print(f"  Final Imbalance Ratio: {imbalance_ratio:.2f}:1")

        print("\n📊 Final Semantic Class Distribution:")
        for class_name, count in sorted(self.final_class_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} instances ({(count/total_instances)*100:.1f}%)")

        self._generate_final_weight_report()

    def _generate_final_weight_report(self):
        total_instances = sum(self.final_class_distribution.values())
        n_classes = len(self.final_class_distribution)
        if total_instances == 0 or n_classes == 0: return
        
        recommended_weights = {cn: total_instances / (n_classes * c) if c > 0 else 0 for cn, c in self.final_class_distribution.items()}
        
        print(f"\n🎯 FINAL SEMANTIC CLASS WEIGHT RECOMMENDATIONS:")
        for class_name, weight in sorted(recommended_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name:<25} → {weight:.4f}")
        
        print("\n💻 PYTORCH EXAMPLE (Semantic):")
        print("```python")
        print("import torch")
        print("import torch.nn as nn")
        print("# Assuming class order is sorted alphabetically for the tensor")
        print("semantic_weight_tensor = torch.tensor([")
        for class_name in sorted(recommended_weights.keys()):
            print(f"    {recommended_weights[class_name]:.4f},  # {class_name}")
        print("], dtype=torch.float32)")
        print("semantic_criterion = nn.CrossEntropyLoss(weight=semantic_weight_tensor)")
        print("```")

    def process_semantic_dataset(self, apply_augmentations=True):
        print("\n" + "="*80)
        print("🎨 STARTING SEMANTIC SEGMENTATION DATASET PROCESSING")
        print("="*80)
        
        self.load_and_analyze_dataset()
        if not self.dataset:
            print("❌ No valid semantic data found. Exiting.")
            return False

        self.apply_instance_split_to_semantic()
        
        if apply_augmentations:
            self.apply_robust_augmentations()
            self.visualize_augmented_samples()
            self.analyze_final_dataset()
            
        print("\n🎉 SEMANTIC SEGMENTATION DATASET PROCESSING COMPLETE!")
        return True


if __name__ == "__main__":
    try:
        import albumentations as A
        print("✅ Albumentations library detected")
    except ImportError:
        print("❌ Albumentations not found! Install with: pip install albumentations")
        exit(1)
    
    # --- Configuration ---
    # WARNING: Setting output directories to be the same as input directories
    # will overwrite the original split data (images and annotations).
    INSTANCE_IMAGES_DIR = "./datasets/instance/images"
    INSTANCE_ANNOTATIONS_DIR = "./datasets/instance/labelme_annotations"
    INSTANCE_OUTPUT_DIR = "./datasets/instance"

    SEMANTIC_IMAGES_DIR = "./datasets/semantic/images"
    SEMANTIC_ANNOTATIONS_DIR = "./datasets/semantic/labelme_annotations"
    SEMANTIC_OUTPUT_DIR = "./datasets/semantic"
    
    # --- Step 1: Process Instance Segmentation Dataset ---
    print("\n" + "="*80)
    print("🎯 STEP 1: PROCESSING INSTANCE SEGMENTATION DATASET")
    print("="*80)
    instance_analyzer = DatasetAnalyzerAndSplitter(
        images_dir=INSTANCE_IMAGES_DIR,
        annotations_dir=INSTANCE_ANNOTATIONS_DIR,
        output_dir=INSTANCE_OUTPUT_DIR
    )
    instance_success, train_files, val_files = instance_analyzer.process_dataset(apply_augmentations=True)
    
    if not instance_success:
        print("❌ Instance dataset processing failed. Halting.")
        exit(1)
    
    # --- Step 2: Process Semantic Segmentation Dataset ---
    print("\n" + "="*80)
    print("🎨 STEP 2: PROCESSING SEMANTIC SEGMENTATION DATASET")
    print("="*80)
    
    semantic_processor = SemanticDatasetProcessor(
        semantic_images_dir=SEMANTIC_IMAGES_DIR,
        semantic_annotations_dir=SEMANTIC_ANNOTATIONS_DIR,
        semantic_output_dir=SEMANTIC_OUTPUT_DIR,
        instance_train_filenames=train_files,
        instance_val_filenames=val_files
    )
    semantic_processor.process_semantic_dataset(apply_augmentations=True)
    
    print("\n" + "="*80)
    print("🎉 DUAL DATASET PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
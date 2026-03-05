# Context-Aware Framework for Bridge Condition Assessment

This repository contains the implementation of a Mask2Former computer vision framework tailored for bridge condition assessment.

## 1. Environment Initialization (WSL / Ubuntu)

The framework utilizes automated setup scripts. To ensure cross-platform compatibility, line endings must be converted prior to execution.

` ` `bash
sudo apt update
sudo apt install dos2unix

# Standardize shell script line endings
dos2unix scripts/*.sh
chmod +x scripts/*.sh

# Execute automated environment construction
./scripts/setup.sh
./scripts/setup_semantic.sh
` ` `

## 2. Mandatory PyTorch AMP Patches

Because the environment dynamically compiles Detectron2 and Mask2Former, critical deprecations in modern PyTorch versions regarding Automatic Mixed Precision (AMP) must be manually patched after running the setup scripts.

**Patch 1: Matcher Module**
* **Target File:** `Mask2Former/mask2former/modeling/matcher.py`
* **Modification:** Locate `with autocast(enabled=False):` and replace it entirely with `with torch.amp.autocast('cuda', enabled=False):`

**Patch 2: Training Loop Engine**
* **Target File:** `mask2former-env/lib/python3.10/site-packages/detectron2/engine/train_loop.py`
* **Modification:** Locate `with autocast(dtype=self.precision):` and replace it entirely with `with torch.amp.autocast('cuda', dtype=self.precision):`

## 3. Execution Pipeline

### 3.1 Dataset Acquisition (Manual)
Due to file size constraints, the proprietary structural dataset is hosted securely on an external server. Prior to executing the pipeline, the dataset must be manually downloaded and extracted.

1. Navigate to the secure repository: https://usu.box.com/s/8el8o8h411f1u0zfg1pzuc22vtam6zid
2. Download the compressed archive to the local machine.
3. Extract the contents directly into the `datasets/` directory located at the root of this project.

### 3.2 Environment Activation
The virtual environment must be activated prior to executing any Python or shell scripts.

` ` `bash
source mask2former-env/bin/activate
` ` `

### 3.3 Data Preprocessing
The preprocessing module must be executed first to format the structural dataset. Ensure class weights are correctly configured within the model parameters before initiating training.

` ` `bash
python dataset_preprocess.py
` ` `

### 3.4 Instance Segmentation

` ` `bash
python datasets.py
./scripts/train.sh
./scripts/eval.sh
` ` `

### 3.5 Semantic Segmentation

` ` `bash
python datasets_semantic.py
./scripts/train_semantic.sh
./scripts/eval_semantic.sh
` ` `

## 4. Analytical Modules

Following evaluation, the analytical modules assess the physical extent of the detected structural damage and quantify model uncertainty.

` ` `bash
./scripts/calculate_corrosion_areas.sh
./scripts/error_analysis.sh 
./scripts/uncertainty_analysis.sh
` ` `

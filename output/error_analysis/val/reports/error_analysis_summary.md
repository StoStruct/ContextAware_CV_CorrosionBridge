# Error Analysis Report (corrosion_val)
## (IoU-Based Instance Matching)

This analysis uses **IoU-based matching** to ensure GT and predicted instances are correctly paired (IoU ≥ 0.5).

## Instance Matching Statistics

| Metric | Value |
|--------|-------|
| Total GT Instances | 360 |
| Total Predicted Instances | 338 |
| Matched Instances | 275 |
| Unmatched GT (False Negatives) | 85 |
| Unmatched Pred (False Positives) | 63 |
| **Recall** | **76.39%** |
| **Precision** | **81.36%** |

## Corrosion Prediction Errors (Matched Instances Only)

| Corrosion Type | Mean Abs Error (%) | Std | Max | Comparisons |
|----------------|--------------------|--------------------|-------------------|--------------|
| Fair | 11.27 | 18.27 | 98.60 | 275 |
| Poor | 8.09 | 18.78 | 98.23 | 275 |
| Severe | 1.01 | 7.90 | 84.26 | 275 |

## Key Findings

- **Best Performance**: Severe (Mean Error: 1.01%)
- **Worst Performance**: Fair (Mean Error: 11.27%)
- **Instance Detection Recall**: 76.39% (model found this % of GT elements)
- **Instance Detection Precision**: 81.36% (this % of predictions matched GT)

## Interpretation

These metrics are calculated **only for correctly matched instances** (IoU ≥ 0.5), providing accurate assessment of corrosion prediction quality.

Unmatched instances represent:
- **False Negatives**: GT elements the model failed to detect
- **False Positives**: Model predictions with no corresponding GT

## Output Files

- `matching_statistics.json`: Instance matching details
- `element_level_errors.csv`: Per-element comparison
- `error_analysis.xlsx`: Comprehensive Excel report with multiple sheets
- `error_analysis_plots.png`: Visualization of errors and matching
- `visualizations/`: Per-image comparison visualizations (2 types per image):
  - `overview_*.png`: 2x2 grid showing semantic + instance results
  - `elementwise_*.png`: Side-by-side GT vs Pred element-wise corrosion (KEY VISUALIZATION!)

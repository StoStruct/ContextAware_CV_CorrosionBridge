# Error Analysis Report (corrosion_test)
## (IoU-Based Instance Matching)

This analysis uses **IoU-based matching** to ensure GT and predicted instances are correctly paired (IoU ≥ 0.5).

## Instance Matching Statistics

| Metric | Value |
|--------|-------|
| Total GT Instances | 66 |
| Total Predicted Instances | 63 |
| Matched Instances | 58 |
| Unmatched GT (False Negatives) | 8 |
| Unmatched Pred (False Positives) | 5 |
| **Recall** | **87.88%** |
| **Precision** | **92.06%** |

## Corrosion Prediction Errors (Matched Instances Only)

| Corrosion Type | Mean Abs Error (%) | Std | Max | Comparisons |
|----------------|--------------------|--------------------|-------------------|--------------|
| Fair | 6.50 | 11.50 | 54.08 | 58 |
| Poor | 1.71 | 2.93 | 12.29 | 58 |
| Severe | 0.20 | 1.45 | 11.14 | 58 |

## Key Findings

- **Best Performance**: Severe (Mean Error: 0.20%)
- **Worst Performance**: Fair (Mean Error: 6.50%)
- **Instance Detection Recall**: 87.88% (model found this % of GT elements)
- **Instance Detection Precision**: 92.06% (this % of predictions matched GT)

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

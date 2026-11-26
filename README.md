# AFM levelling in Python

Week 7 project update by Harrison Ouyang. This repository contains the Python implementation and preliminary results for **Case 3**, a synthetic atomic force microscopy (AFM) benchmark. The clean surface contains raised objects and genuine substrate corrugation. The observed input adds a known planar tilt and a fixed noise field. The aim is to remove the injected tilt while preserving the topography used to judge object shape.

## Start here

The [English Week 7 report](Week7_Report_English.pdf) explains the experimental design on pages 2–3, the comparisons on pages 4–7, and the preliminary conclusions and limitations on pages 8–9.

- **Python versus Gwyddion** checks whether four matched classical methods produce the same spatial correction on the same input. The pixelwise comparison allows one constant height offset to account for different zero-height conventions.
- **Each output versus the clean target** tests whether the correction works for this surface. The removed field, residual and fixed profile show where tilt remains or real features change. Agreement between the two implementations can coexist with an error relative to the clean target.

## Code

| File | What it does |
|---|---|
| [`week7_case03.py`](code/week7_case03.py) | Reads calibrated height arrays from the Gwyddion archive, runs classical fits and the two pretrained afMLevel workflows, and exports the numerical comparisons. |
| [`week7_figures.py`](code/week7_figures.py) | Generates height, removal and residual maps with shared scales. It also exports the fixed profile and metric summary. |
| [`requirements.txt`](code/requirements.txt) | Lists the Python package versions used for the benchmark. |

## Figures

| Figure | What to look for |
|---|---|
| [01 · Classical methods versus the clean target](figures/01_classic_vs_clean.png) | Read down each column from corrected height to total removal to residual. The left column is the clean target, known tilt and zero error. The Gwyddion Flatten Base column is an archived reference. |
| [02 · Masked and learned methods versus the clean target](figures/02_masks_and_ml_vs_clean.png) | Compare the oracle- and learned-mask plane fits, then the two full afMLevel pipelines. The scales match Figure 01. |
| [03 · Python versus Gwyddion](figures/03_python_vs_gwyddion.png) | Compare the paired height maps in the first two rows. The bottom row shows the pixelwise difference after constant alignment. |
| [04 · Processing stages and fit pixels](figures/04_stage_evidence.png) | Follow the first removal, the pixels selected for fitting and the later removal. White pixels in the mask panels are included in the fit. |
| [05 · Fixed height profile](figures/05_fixed_profile.png) | Compare the same line at y = 2.500 µm across methods. The upper panels show the two object peaks and surrounding substrate. The lower panels show the residual along that line. |
| [06 · Summary metrics](figures/06_metrics_summary.png) | Read residual RMSE alongside the corrugation and remaining-tilt coefficients. A small remaining tilt can coincide with loss of genuine corrugation. |

## Tables

| File | What it reports |
|---|---|
| [`python_vs_gwyddion.csv`](tables/python_vs_gwyddion.csv) | Mean height difference before alignment and pixelwise error after alignment for each matched method. Use this table to check reproduction of the Gwyddion output. |
| [`metrics_vs_clean.csv`](tables/metrics_vs_clean.csv) | Clean-target residual RMSE for the full field, foreground and substrate. It also reports the removal range, removal error relative to known tilt, and fitted coefficients for objects, corrugation and remaining tilt. |
| [`profile_truth_peak_errors.csv`](tables/profile_truth_peak_errors.csv) | Signed height errors at two fixed positions defined by the clean object peaks. Positive and negative values indicate points above and below the clean target, respectively. |

## Feedback requested

Is a first-order fit a sensible baseline for the known planar tilt in Case 3? Would you change how mask selection, polynomial order, or the full afMLevel routines are compared before extending the study to other cases?

The report, figures and tables are ready to review here. Rerunning the benchmark requires the original Case 3 Gwyddion archive, which is available separately along with the large intermediate arrays.

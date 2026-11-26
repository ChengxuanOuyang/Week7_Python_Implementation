# AFM levelling in Python

Week 7 project update by Harrison Ouyang. The current benchmark uses one synthetic AFM image with known object shapes, substrate corrugation, tilt and noise.

## Start here

- [Week 7 report](Week7_Report_English.pdf) covers the experiment, results and limitations.
- [Python versus Gwyddion](figures/03_python_vs_gwyddion.png) checks implementation agreement for four matched classical methods.
- [Classical methods versus the clean target](figures/01_classic_vs_clean.png) and [masked and learned methods versus the clean target](figures/02_masks_and_ml_vs_clean.png) show the corrected heights, removed backgrounds and residuals.
- [Processing stages](figures/04_stage_evidence.png) and the [fixed height profile](figures/05_fixed_profile.png) show which steps alter the topography.
- [Results tables](tables/) and [Python code](code/) provide the numerical evidence and implementation.

## Feedback requested

Is a first-order fit a sensible baseline for the known planar tilt in Case 3? Would you change how mask selection, polynomial order, or the full afMLevel routines are compared before extending the study to other cases?

The reports and results can be reviewed directly from this repository. Running the scripts requires the original Case 3 Gwyddion archive. The full lab archive and large intermediate arrays are available separately.

==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-02T15:25:43.894655
Model: phi3
Directory: data
Pattern: task_*.json

RESULTS:
  Total tasks:       140
  Successful:        140 (100.0%)
  Correct (100%):    102 (72.9%)
  Failed:            0

ACCURACY:
  Overall:           96.1%
  When successful:   96.1%

TIMING:
  Total time:        1583.5s
  Avg per task:      11.3s
  Avg LLM time:      11.145s
  Avg detection:     0.002s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    83.6%
  LLM success rate:  60.9%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  flood_fill: 100.0% (9/9)
  tiling: 100.0% (20/20)
  scaling: 99.7% (10/11)
  draw_line: 99.6% (8/9)
  translation: 99.2% (14/18)
  symmetry: 97.9% (6/10)
  composite: 97.8% (10/14)
  reflection: 96.8% (13/17)
  rotation: 92.8% (6/17)
  color_change: 73.5% (2/4)

TRANSFORMATION COVERAGE: 11/11 (100%)
  Tested: add_border, color_change, composite, draw_line, flood_fill, reflection, rotation, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-02T10:11:45.725533
Model: llama3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       64
  Successful:        64 (100.0%)
  Correct (100%):    55 (85.9%)
  Failed:            0

ACCURACY:
  Overall:           98.1%
  When successful:   98.1%

TIMING:
  Total time:        387.2s
  Avg per task:      6.0s
  Avg LLM time:      5.512s
  Avg detection:     0.001s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    37.5%
  LLM success rate:  90.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (2/2)
  translation: 100.0% (11/11)
  draw_line: 100.0% (5/5)
  flood_fill: 100.0% (4/4)
  tiling: 100.0% (5/5)
  scaling: 100.0% (4/4)
  symmetry: 100.0% (4/4)
  rotation: 96.6% (4/6)
  reflection: 94.3% (6/8)
  composite: 93.0% (3/8)

TRANSFORMATION COVERAGE: 11/11 (100%)
  Tested: add_border, color_change, composite, draw_line, flood_fill, reflection, rotation, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
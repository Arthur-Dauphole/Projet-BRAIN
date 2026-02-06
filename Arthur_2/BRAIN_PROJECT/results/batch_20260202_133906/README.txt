==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-02T13:39:06.276656
Model: llama3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       140
  Successful:        140 (100.0%)
  Correct (100%):    99 (70.7%)
  Failed:            0

ACCURACY:
  Overall:           96.8%
  When successful:   96.8%

TIMING:
  Total time:        723.0s
  Avg per task:      5.2s
  Avg LLM time:      4.816s
  Avg detection:     0.001s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    34.3%
  LLM success rate:  73.9%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (4/4)
  flood_fill: 100.0% (9/9)
  tiling: 100.0% (20/20)
  draw_line: 99.6% (8/9)
  translation: 99.2% (14/18)
  scaling: 99.0% (9/11)
  symmetry: 97.9% (6/10)
  reflection: 96.1% (12/17)
  rotation: 92.8% (5/17)
  composite: 92.7% (3/14)

TRANSFORMATION COVERAGE: 11/11 (100%)
  Tested: add_border, color_change, composite, draw_line, flood_fill, reflection, rotation, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
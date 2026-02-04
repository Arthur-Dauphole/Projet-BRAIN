==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-04T09:12:17.985458
Model: gemma2
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       140
  Successful:        140 (100.0%)
  Correct (100%):    116 (82.9%)
  Failed:            0

ACCURACY:
  Overall:           97.7%
  When successful:   97.7%

TIMING:
  Total time:        890.3s
  Avg per task:      6.4s
  Avg LLM time:      6.091s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    61.4%
  LLM success rate:  92.6%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  draw_line: 100.0% (10/10)
  flood_fill: 100.0% (10/10)
  tiling: 100.0% (20/20)
  scaling: 99.7% (10/11)
  symmetry: 99.3% (8/10)
  translation: 99.2% (14/18)
  color_change: 98.5% (3/4)
  composite: 97.8% (10/14)
  reflection: 96.8% (13/16)
  rotation: 94.7% (10/16)

TRANSFORMATION COVERAGE: 11/11 (100%)
  Tested: add_border, color_change, composite, draw_line, flood_fill, reflection, rotation, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
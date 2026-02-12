==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-09T14:25:45.727308
Model: llama3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       50
  Successful:        50 (100.0%)
  Correct (100%):    35 (70.0%)
  Failed:            0

ACCURACY:
  Overall:           97.0%
  When successful:   97.0%

TIMING:
  Total time:        328.7s
  Avg per task:      6.6s
  Avg LLM time:      6.561s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    62.0%
  LLM success rate:  94.7%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (3/3)
  draw_line: 100.0% (10/10)
  translation: 99.0% (5/8)
  composite: 97.7% (9/13)
  reflection: 95.5% (2/4)
  rotation: 88.7% (2/7)

TRANSFORMATION COVERAGE: 7/11 (64%)
  Tested: add_border, color_change, composite, draw_line, reflection, rotation, translation
  Not tested: flood_fill, scaling, symmetry, tiling

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
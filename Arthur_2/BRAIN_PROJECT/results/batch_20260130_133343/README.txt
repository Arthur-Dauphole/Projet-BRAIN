==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-30T13:33:43.146682
Model: mistral
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       52
  Successful:        52 (100.0%)
  Correct (100%):    43 (82.7%)
  Failed:            0

ACCURACY:
  Overall:           98.0%
  When successful:   98.0%

TIMING:
  Total time:        366.4s
  Avg per task:      7.0s
  Avg LLM time:      6.575s
  Avg detection:     0.001s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    23.1%
  LLM success rate:  90.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (2/2)
  translation: 100.0% (11/11)
  draw_line: 100.0% (5/5)
  tiling: 100.0% (5/5)
  reflection: 98.4% (7/8)
  rotation: 96.6% (4/6)
  composite: 93.0% (3/8)

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
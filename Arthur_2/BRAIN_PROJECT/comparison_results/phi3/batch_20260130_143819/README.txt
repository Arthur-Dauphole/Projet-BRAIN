==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-30T14:38:19.918721
Model: phi3
Directory: data
Pattern: task_*.json

RESULTS:
  Total tasks:       52
  Successful:        52 (100.0%)
  Correct (100%):    39 (75.0%)
  Failed:            0

ACCURACY:
  Overall:           90.0%
  When successful:   90.0%

TIMING:
  Total time:        385.2s
  Avg per task:      7.4s
  Avg LLM time:      5.921s
  Avg detection:     0.001s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    23.1%
  LLM success rate:  80.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (2/2)
  draw_line: 100.0% (5/5)
  reflection: 98.4% (7/8)
  composite: 93.0% (3/8)
  translation: 81.8% (9/11)
  rotation: 81.5% (4/6)
  tiling: 80.0% (4/5)

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
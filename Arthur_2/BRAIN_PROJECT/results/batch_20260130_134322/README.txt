==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-30T13:43:22.404085
Model: phi3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       52
  Successful:        52 (100.0%)
  Correct (100%):    42 (80.8%)
  Failed:            0

ACCURACY:
  Overall:           97.5%
  When successful:   97.5%

TIMING:
  Total time:        270.9s
  Avg per task:      5.2s
  Avg LLM time:      5.110s
  Avg detection:     0.000s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    26.9%
  LLM success rate:  86.8%

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
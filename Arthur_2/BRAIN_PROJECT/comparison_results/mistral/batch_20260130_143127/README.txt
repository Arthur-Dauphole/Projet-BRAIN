==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-30T14:31:27.582865
Model: mistral
Directory: data
Pattern: task_*.json

RESULTS:
  Total tasks:       52
  Successful:        52 (100.0%)
  Correct (100%):    41 (78.8%)
  Failed:            0

ACCURACY:
  Overall:           97.6%
  When successful:   97.6%

TIMING:
  Total time:        387.6s
  Avg per task:      7.5s
  Avg LLM time:      6.929s
  Avg detection:     0.001s
  Avg execution:     0.000s

LLM vs FALLBACK:
  Fallback usage:    23.1%
  LLM success rate:  85.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (2/2)
  translation: 100.0% (11/11)
  draw_line: 100.0% (5/5)
  tiling: 100.0% (5/5)
  reflection: 97.7% (6/8)
  rotation: 96.6% (4/6)
  composite: 93.0% (3/8)

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
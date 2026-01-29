==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-29T10:59:21.463232
Model: llama3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       16
  Successful:        16 (100.0%)
  Correct (100%):    14 (87.5%)
  Failed:            0

ACCURACY:
  Overall:           97.5%
  When successful:   97.5%

TIMING:
  Total time:        114.6s
  Avg per task:      7.2s
  Avg LLM time:      6.240s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    25.0%
  LLM success rate:  91.7%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (1/1)
  color_change: 100.0% (1/1)
  rotation: 100.0% (1/1)
  translation: 100.0% (2/2)
  draw_line: 100.0% (1/1)
  tiling: 100.0% (2/2)
  composite: 97.9% (2/3)
  reflection: 83.3% (1/2)

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
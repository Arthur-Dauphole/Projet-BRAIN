==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.10.0
Run timestamp: 2026-01-30T14:21:33.497830
Model: llama3
Directory: data
Pattern: task_*.json

RESULTS:
  Total tasks:       10
  Successful:        10 (100.0%)
  Correct (100%):    9 (90.0%)
  Failed:            0

ACCURACY:
  Overall:           99.4%
  When successful:   99.4%

TIMING:
  Total time:        71.9s
  Avg per task:      7.2s
  Avg LLM time:      6.359s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    50.0%
  LLM success rate:  100.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  color_change: 100.0% (1/1)
  reflection: 100.0% (1/1)
  rotation: 100.0% (1/1)
  translation: 100.0% (1/1)
  composite: 93.8% (0/1)

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
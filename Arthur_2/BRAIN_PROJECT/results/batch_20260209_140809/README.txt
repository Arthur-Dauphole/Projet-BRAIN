==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-09T14:08:09.894821
Model: llama3
Directory: data/
Pattern: task_*.json

RESULTS:
  Total tasks:       10
  Successful:        10 (100.0%)
  Correct (100%):    4 (40.0%)
  Failed:            0

ACCURACY:
  Overall:           90.5%
  When successful:   90.5%

TIMING:
  Total time:        84.6s
  Avg per task:      8.5s
  Avg LLM time:      8.410s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    90.0%
  LLM success rate:  0.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  rotation: 84.1% (0/5)

TRANSFORMATION COVERAGE: 2/11 (18%)
  Tested: add_border, rotation
  Not tested: color_change, composite, draw_line, flood_fill, reflection, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
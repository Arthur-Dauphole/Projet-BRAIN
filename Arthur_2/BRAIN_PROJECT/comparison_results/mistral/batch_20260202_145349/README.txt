==================================================
BRAIN Batch Evaluation Results
==================================================

Program version: 1.11.0
Run timestamp: 2026-02-02T14:53:49.255664
Model: mistral
Directory: data
Pattern: task_*.json

RESULTS:
  Total tasks:       140
  Successful:        140 (100.0%)
  Correct (100%):    103 (73.6%)
  Failed:            0

ACCURACY:
  Overall:           97.5%
  When successful:   97.5%

TIMING:
  Total time:        1851.5s
  Avg per task:      13.2s
  Avg LLM time:      6.323s
  Avg detection:     0.001s
  Avg execution:     0.001s

LLM vs FALLBACK:
  Fallback usage:    62.1%
  LLM success rate:  83.0%

ACCURACY BY TRANSFORMATION:
  add_border: 100.0% (4/4)
  flood_fill: 100.0% (9/9)
  tiling: 100.0% (20/20)
  scaling: 99.7% (10/11)
  draw_line: 99.6% (8/9)
  translation: 99.2% (14/18)
  color_change: 98.5% (3/4)
  symmetry: 97.9% (6/10)
  composite: 97.8% (10/14)
  reflection: 96.8% (13/17)
  rotation: 92.8% (6/17)

TRANSFORMATION COVERAGE: 11/11 (100%)
  Tested: add_border, color_change, composite, draw_line, flood_fill, reflection, rotation, scaling, symmetry, tiling, translation

FILES:
  summary.json  - Full detailed report (JSON)
  tasks.csv     - Task-level results for data analysis
  images/       - Visual results

==================================================
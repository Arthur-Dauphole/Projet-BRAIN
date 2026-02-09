# BRAIN Project - Analysis Report

**Generated:** 2026-02-04 11:39:42

## Overview

- **Total tasks analyzed:** 1356
- **Overall accuracy:** 97.6%
- **Success rate (100%):** 80.9%

## Performance by Model

| Model | Mean Accuracy | Std | N |
|-------|---------------|-----|---|
| gemma2 | 97.8% | 0.094 | 840 |
| llama3 | 97.3% | 0.086 | 412 |
| mistral | 98.0% | 0.048 | 52 |
| phi3 | 97.5% | 0.057 | 52 |

## Performance by Transformation

| Transformation | Mean Accuracy | Success Rate | N |
|----------------|---------------|--------------|---|
| add_border | 100.0% | 100.0% | 49 |
| flood_fill | 100.0% | 100.0% | 81 |
| tiling | 100.0% | 100.0% | 182 |
| draw_line | 99.9% | 96.9% | 98 |
| scaling | 99.7% | 90.2% | 92 |
| translation | 99.4% | 83.2% | 190 |
| symmetry | 99.0% | 76.2% | 84 |
| color_change | 99.0% | 82.9% | 41 |
| reflection | 96.7% | 79.4% | 165 |
| composite | 96.3% | 59.2% | 147 |
| rotation | 94.4% | 53.8% | 156 |

## LLM vs Fallback

- **Fallback usage rate:** 52.9%
- **LLM-only accuracy:** 97.8%
- **With fallback accuracy:** 97.5%

# BRAIN Project - Analysis Report

**Generated:** 2026-02-02 16:02:31

## Overview

- **Total tasks analyzed:** 516
- **Overall accuracy:** 97.4%
- **Success rate (100%):** 78.1%

## Performance by Model

| Model | Mean Accuracy | Std | N |
|-------|---------------|-----|---|
| llama3 | 97.3% | 0.086 | 412 |
| mistral | 98.0% | 0.048 | 52 |
| phi3 | 97.5% | 0.057 | 52 |

## Performance by Transformation

| Transformation | Mean Accuracy | Success Rate | N |
|----------------|---------------|--------------|---|
| add_border | 100.0% | 100.0% | 25 |
| flood_fill | 100.0% | 100.0% | 22 |
| tiling | 100.0% | 100.0% | 62 |
| draw_line | 99.8% | 94.9% | 39 |
| translation | 99.7% | 90.2% | 82 |
| color_change | 99.6% | 94.1% | 17 |
| scaling | 99.5% | 88.5% | 26 |
| symmetry | 98.3% | 66.7% | 24 |
| reflection | 96.5% | 77.9% | 68 |
| rotation | 94.5% | 47.5% | 59 |
| composite | 94.3% | 42.9% | 63 |

## LLM vs Fallback

- **Fallback usage rate:** 39.0%
- **LLM-only accuracy:** 97.7%
- **With fallback accuracy:** 97.0%

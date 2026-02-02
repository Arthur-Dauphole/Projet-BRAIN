# BRAIN Project - Analysis Report

**Generated:** 2026-01-30 15:50:10

## Overview

- **Total tasks analyzed:** 172
- **Overall accuracy:** 97.7%
- **Success rate (100%):** 82.0%

## Performance by Model

| Model | Mean Accuracy | Std | N |
|-------|---------------|-----|---|
| llama3 | 97.6% | 0.062 | 68 |
| mistral | 98.0% | 0.048 | 52 |
| phi3 | 97.5% | 0.057 | 52 |

## Performance by Transformation

| Transformation | Mean Accuracy | Success Rate | N |
|----------------|---------------|--------------|---|
| add_border | 100.0% | 100.0% | 13 |
| color_change | 100.0% | 100.0% | 7 |
| draw_line | 100.0% | 100.0% | 16 |
| tiling | 100.0% | 100.0% | 17 |
| translation | 100.0% | 100.0% | 35 |
| reflection | 97.3% | 84.6% | 26 |
| rotation | 96.8% | 68.4% | 19 |
| composite | 93.6% | 40.7% | 27 |

## LLM vs Fallback

- **Fallback usage rate:** 24.4%
- **LLM-only accuracy:** 98.3%
- **With fallback accuracy:** 95.9%

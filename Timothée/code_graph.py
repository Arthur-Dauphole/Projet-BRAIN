#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_ieee_figures.py

Generates 7 publication‑quality figures (IEEE style) from the JSON statistics file.
All text is in English, using the same style as visualizer.py for perfect consistency.
Output: vector PDF files in the 'figures/' folder.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# IEEE PUBLICATION STYLE CONFIGURATION (from visualizer.py)
# =============================================================================

def _setup_publication_style():
    """Configure matplotlib for IEEE‑style publication figures."""
    latex_available = shutil.which('latex') is not None

    plt.rcParams.update({
        # LaTeX rendering (only if available)
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if latex_available else ["DejaVu Serif"],
        
        # Font sizes (IEEE standard, slightly reduced for titles)
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,          # reduced from 11 → better balance
        "legend.fontsize": 7,
        "xtick.labelsize": 6,         # smaller for long labels
        "ytick.labelsize": 7,
        
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "savefig.pad_inches": 0.02,
        
        # Axes settings
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
    return latex_available

# Initialize style
LATEX_AVAILABLE = _setup_publication_style()

# =============================================================================
# STYLE CONSTANTS (from visualizer.py)
# =============================================================================

# IEEE single‑column width: 3.5 inches (perfect for double‑column documents)
IEEE_SINGLE_COLUMN = (3.5, 2.5)

# Colorblind‑friendly palette (Wong, 2011)
COLORS = {
    "primary": "#0072B2",     # blue
    "secondary": "#009E73",   # green
    "accent": "#E69F00",      # orange
    "error": "#D55E00",       # vermillion
    "neutral": "#999999",     # grey
    "purple": "#CC79A7",      # reddish purple
    "yellow": "#F0E442",      # yellow
    "black": "#000000",
}

# Colors for rule types (matching TRANSFORM_COLORS in visualizer.py)
RULE_TYPE_COLORS = {
    "translations": COLORS["primary"],      # blue
    "color_changes": COLORS["accent"],      # orange
    "connections": COLORS["purple"],        # purple
}

# Colors for rule combinations (max 5 categories)
COMBO_COLORS = [
    COLORS["primary"],      # blue
    COLORS["secondary"],    # green
    COLORS["accent"],       # orange
    COLORS["error"],        # vermillion
    COLORS["purple"],       # purple
]

# =============================================================================
# DATA LOADING
# =============================================================================

JSON_PATH = "statistics_20260210_084609.json"
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SAVE FUNCTION (identical to visualizer.py)
# =============================================================================

def save_figure(fig: plt.Figure, filename: str, formats: List[str] = None):
    """
    Save a figure in the specified formats.
    
    Args:
        fig: matplotlib Figure object
        filename: base name (without extension)
        formats: list of extensions (default: ["pdf"])
    """
    if formats is None:
        formats = ["pdf"]
    base_path = OUTPUT_DIR / filename
    for fmt in formats:
        filepath = base_path.with_suffix(f".{fmt}")
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")

# =============================================================================
# FIGURE 1 : GLOBAL ACCURACY
# =============================================================================

def plot_overall_accuracy():
    """Single bar: overall accuracy."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    acc = data['overall_accuracy'] * 100
    ax.bar(['Global'], [acc], color=COLORS["primary"], width=0.6,
           edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Global accuracy')
    ax.text(0, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)
    
    save_figure(fig, "01_overall_accuracy")
    plt.close(fig)

# =============================================================================
# FIGURE 2 : RULE TYPE DISTRIBUTION
# =============================================================================

def plot_rule_type_distribution():
    """Number of files per rule type."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    rule_dist = data['rule_type_distribution']
    rules = list(rule_dist.keys())
    counts = list(rule_dist.values())
    colors = [RULE_TYPE_COLORS.get(r, COLORS["neutral"]) for r in rules]
    
    x = np.arange(len(rules))
    bars = ax.bar(x, counts, color=colors, width=0.6,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=6)
    ax.set_ylabel('Number of files')
    ax.set_title('Rule type presence')
    
    # Annotations: slightly higher offset to avoid crowding
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(v), ha='center', va='bottom', fontsize=6)
    
    save_figure(fig, "02_rule_type_distribution")
    plt.close(fig)

# =============================================================================
# FIGURE 3 : ACCURACY BY RULE TYPE
# =============================================================================

def plot_rule_type_accuracy():
    """Average accuracy per rule type."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    rule_acc = data['rule_type_accuracy']
    rules = list(rule_acc.keys())
    acc_vals = [rule_acc[r] * 100 for r in rules]
    colors = [RULE_TYPE_COLORS.get(r, COLORS["neutral"]) for r in rules]
    
    x = np.arange(len(rules))
    bars = ax.bar(x, acc_vals, color=colors, width=0.6,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by rule type')
    
    for bar, v in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.8,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=6)
    
    save_figure(fig, "03_rule_type_accuracy")
    plt.close(fig)

# =============================================================================
# FIGURE 4 : RULE COMBINATION DISTRIBUTION
# =============================================================================

def plot_rule_combinations():
    """Frequency of rule combinations."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    combos = data['rule_combinations']
    # Sort descending
    sorted_items = sorted(combos.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    x = np.arange(len(labels))
    colors = [COMBO_COLORS[i % len(COMBO_COLORS)] for i in range(len(labels))]
    
    bars = ax.bar(x, counts, color=colors, width=0.6,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    # Increased rotation to 45°, smaller font, adjusted alignment
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5.5)
    ax.set_ylabel('Number of files')
    ax.set_title('Rule combinations')
    
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', va='bottom', fontsize=6)
    
    save_figure(fig, "04_rule_combinations")
    plt.close(fig)

# =============================================================================
# TASK CATEGORIES PREPARATION
# =============================================================================

categories = defaultdict(lambda: {'total': 0, 'correct': 0})
for item in data['individual_results']:
    fname = item['filename']
    parts = fname.split('_')
    cat = parts[1] if len(parts) >= 2 else 'other'
    categories[cat]['total'] += 1
    if item['correct_tests'] > 0:
        categories[cat]['correct'] += 1

sorted_cats = sorted(categories.keys())
cat_names = []
cat_totals = []
cat_acc = []
for cat in sorted_cats:
    cat_names.append(cat)
    tot = categories[cat]['total']
    corr = categories[cat]['correct']
    cat_totals.append(tot)
    cat_acc.append(corr / tot * 100 if tot > 0 else 0)

# =============================================================================
# FIGURE 5 : NUMBER OF FILES PER CATEGORY
# =============================================================================

def plot_category_counts():
    """File distribution by task category."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    x = np.arange(len(cat_names))
    bars = ax.bar(x, cat_totals, color=COLORS["primary"], width=0.6,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    # Increased rotation to 45°, smaller font
    ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=5.5)
    ax.set_ylabel('Number of files')
    ax.set_title('Distribution by category')
    
    for bar, v in zip(bars, cat_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', va='bottom', fontsize=6)
    
    save_figure(fig, "05_category_counts")
    plt.close(fig)

# =============================================================================
# FIGURE 6 : ACCURACY PER CATEGORY
# =============================================================================

def plot_category_accuracy():
    """Average accuracy per task category."""
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    x = np.arange(len(cat_names))
    bars = ax.bar(x, cat_acc, color=COLORS["primary"], width=0.6,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=5.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by category')
    
    for bar, v in zip(bars, cat_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.8,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=6)
    
    save_figure(fig, "06_category_accuracy")
    plt.close(fig)

# =============================================================================
# FIGURE 7 : EXECUTION TIME DISTRIBUTION
# =============================================================================

def plot_execution_time_distribution():
    """
    Histogram of individual execution times.
    Includes a vertical line for the global mean.
    """
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COLUMN, layout='constrained')
    
    # Extract all individual execution times
    exec_times = [item['execution_time'] for item in data['individual_results']]
    mean_time = np.mean(exec_times)
    median_time = np.median(exec_times)
    
    # Histogram
    n, bins_edges, patches = ax.hist(exec_times, bins=20, 
                                      color=COLORS["primary"], 
                                      edgecolor='black', linewidth=0.5,
                                      alpha=0.8)
    
    # Vertical line for the mean
    ax.axvline(mean_time, color=COLORS["error"], linestyle='--', 
               linewidth=1.2, label=f'Mean = {mean_time:.4f} s')
    
    # Median annotation placed at top-left to avoid overlap
    ax.text(0.02, 0.95, f'Median = {median_time:.4f} s',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=6, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Execution time (s)')
    ax.set_ylabel('Number of files')
    ax.set_title('Distribution of execution times')
    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlim(0, max(exec_times) * 1.05)
    
    save_figure(fig, "07_execution_time_distribution")
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Generating 7 publication‑quality figures (English, IEEE style)...")
    plot_overall_accuracy()
    plot_rule_type_distribution()
    plot_rule_type_accuracy()
    plot_rule_combinations()
    plot_category_counts()
    plot_category_accuracy()
    plot_execution_time_distribution()
    print(f"\n✅ All figures saved to: {OUTPUT_DIR}/")
    print("   Format: vector PDF (ready for LaTeX inclusion).")
"""
LaTeX-ready (2-column) figure
Accuracy vs Electricity Cost – Apple M1

Features:
- Single-column size
- Times-like typography
- Direct model labels (no legend)
- Pareto frontier visualization
- Dominated point annotation
- Sweet spot highlighting
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =============================================================================
# 1. FIGURE CONFIG (2-column paper)
# =============================================================================

COLUMN_WIDTH = 3.4
FIG_HEIGHT = 2.6
OUTPUT_NAME = "accuracy_cost_m1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

COLORS = {
    "Mistral 7B": "#1f77b4",
    "Llama 3": "#2ca02c",
    "Gemma 2": "#d62728",
    "Phi 3": "#9467bd"
}

MARKERS = {
    "Mistral 7B": "o",
    "Llama 3": "s",
    "Gemma 2": "^",
    "Phi 3": "D"
}

# =============================================================================
# 2. DATA
# =============================================================================

M1_POWER_WATTS = 30.0
TOTAL_TASKS = 140
PRICE_EUR_PER_KWH = 0.27


def load_data():
    data = {
        "Model": ["Mistral 7B", "Llama 3", "Gemma 2", "Phi 3"],
        "Correct": [101, 99, 97, 94],
        "Time_ms_avg": [14844, 11608, 15083, 8671],
    }
    return pd.DataFrame(data)


def compute_metrics(df):
    df = df.copy()
    df["Accuracy"] = (df["Correct"] / TOTAL_TASKS) * 100
    df["Time_hours"] = df["Time_ms_avg"] / 1000 / 3600

    df["Cost_1k"] = (
        (M1_POWER_WATTS / 1000)
        * df["Time_hours"]
        * PRICE_EUR_PER_KWH
        * 1000
    )
    return df


# =============================================================================
# 3. PLOT
# =============================================================================

def plot_figure(df):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT))

    # Light grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # =========================================================================
    # NEW: Sweet Spot Zone (around Llama 3)
    # =========================================================================
    llama_row = df[df["Model"] == "Llama 3"].iloc[0]
    llama_x = llama_row["Cost_1k"]
    llama_y = llama_row["Accuracy"]
    
    # Create elliptical sweet spot
    from matplotlib.patches import Ellipse
    sweet_spot = Ellipse(
        xy=(llama_x, llama_y),
        width=0.0030,  # Adjust for X spread
        height=2.5,     # Adjust for Y spread
        facecolor='gold',
        alpha=0.15,
        edgecolor='orange',
        linewidth=0.8,
        linestyle='--',
        zorder=1
    )
    ax.add_patch(sweet_spot)

    # =========================================================================
    # NEW: Pareto Frontier (Phi 3 → Llama 3 → Mistral 7B)
    # =========================================================================
    pareto_models = ["Phi 3", "Llama 3", "Mistral 7B"]
    pareto_points = df[df["Model"].isin(pareto_models)].sort_values("Cost_1k")
    
    ax.plot(
        pareto_points["Cost_1k"],
        pareto_points["Accuracy"],
        color='gray',
        linestyle='--',
        linewidth=1.2,
        alpha=0.6,
        zorder=2,
        label='Pareto frontier'
    )

    # Scatter points
    for _, row in df.iterrows():
        model = row["Model"]
        x = row["Cost_1k"]
        y = row["Accuracy"]

        # Scatter point
        ax.scatter(
            x,
            y,
            s=40,
            color=COLORS[model],
            marker=MARKERS[model],
            edgecolors="black",
            linewidth=0.4,
            zorder=3
        )

        # Direct label above point (bold for readability)
        ax.text(
            x,
            y + 0.8,
            model,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=4
        )

    # =========================================================================
    # NEW: Dominated Point Annotation (Gemma 2)
    # =========================================================================
    gemma_row = df[df["Model"] == "Gemma 2"].iloc[0]
    gemma_x = gemma_row["Cost_1k"]
    gemma_y = gemma_row["Accuracy"]
    
    ax.annotate(
        'Dominated',
        xy=(gemma_x + 0.0001, gemma_y - 0.0001),
        xytext=(gemma_x + 0.0015, gemma_y - 1.5),
        fontsize=6,
        color='darkred',
        style='italic',
        ha='left',
        va='top',
        arrowprops=dict(
            arrowstyle='->',
            color='darkred',
            linewidth=0.8,
            alpha=0.7
        ),
        zorder=5
    )

    # Small note indicating zoom
    ax.text(
        0.02, 0.02,
        "Y-axis truncated",
        transform=ax.transAxes,
        fontsize=6,
        ha="left",
        va="bottom"
    )

    ax.set_title(
        "Accuracy vs Electricity Cost (Apple M1)",
        fontsize=9,
        pad=4
    )

    # Axis labels
    ax.set_xlabel("Electricity cost (€/1k tasks)")
    ax.set_ylabel("Accuracy (%)")

    # Scientific transparency: full range
    ax.set_ylim(65, 80)

    # Tight X limits
    ax.set_xlim(df["Cost_1k"].min() * 0.9, df["Cost_1k"].max() * 1.05)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.savefig(f"{OUTPUT_NAME}.png")
    plt.close()
    
    print(f"✅ Figure saved: {OUTPUT_NAME}.pdf and {OUTPUT_NAME}.png")


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    df = load_data()
    df = compute_metrics(df)
    plot_figure(df)
    
    # Print summary for paper
    print("\n📊 Summary Statistics:")
    print(df[["Model", "Accuracy", "Cost_1k"]].to_string(index=False))


if __name__ == "__main__":
    main()
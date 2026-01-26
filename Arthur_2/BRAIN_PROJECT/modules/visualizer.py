"""
visualizer.py - Dashboard Visualizer (Visualization Module)
===========================================================
Step 5 of the pipeline: Visualization

Provides graphical visualization of grids, predictions,
and analysis results using matplotlib.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# Lazy import matplotlib to avoid issues if not installed
plt = None
colors_module = None


def _ensure_matplotlib():
    """Lazy import matplotlib."""
    global plt, colors_module
    if plt is None:
        import matplotlib.pyplot as plt_import
        import matplotlib.colors as colors_import
        plt = plt_import
        colors_module = colors_import


# ARC color palette (official colors)
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Azure
    "#870C25",  # 9: Brown
]


class Visualizer:
    """
    Visualization dashboard for ARC grids and results.
    
    Responsibilities:
        - Display individual grids
        - Show input-output pairs
        - Visualize predictions vs expected
        - Highlight differences
        - Generate analysis dashboards
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Dots per inch for figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self._cmap = None
    
    def _get_cmap(self):
        """Get or create the ARC colormap."""
        if self._cmap is None:
            _ensure_matplotlib()
            self._cmap = colors_module.ListedColormap(ARC_COLORS)
        return self._cmap
    
    def show_grid(
        self, 
        grid, 
        title: str = "Grid",
        show_gridlines: bool = True,
        ax=None
    ):
        """
        Display a single grid.
        
        Args:
            grid: Grid object or 2D numpy array
            title: Title for the plot
            show_gridlines: Whether to show cell borders
            ax: Optional matplotlib axes to plot on
        """
        _ensure_matplotlib()
        
        # Get data as numpy array
        if hasattr(grid, 'data'):
            data = grid.data
        else:
            data = np.array(grid)
        
        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the grid
        im = ax.imshow(
            data, 
            cmap=self._get_cmap(), 
            vmin=0, 
            vmax=9,
            interpolation='nearest'
        )
        
        # Add gridlines
        if show_gridlines:
            ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=0.5)
        
        # Remove major ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(title)
        
        return ax
    
    def show_pair(
        self, 
        input_grid, 
        output_grid, 
        title: str = "Input → Output"
    ):
        """
        Display an input-output pair side by side.
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            title: Overall title
        """
        _ensure_matplotlib()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        self.show_grid(input_grid, "Input", ax=axes[0])
        self.show_grid(output_grid, "Output", ax=axes[1])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def show_comparison(
        self, 
        input_grid, 
        predicted_grid, 
        expected_grid,
        title: str = "Prediction Comparison"
    ):
        """
        Display input, predicted output, and expected output.
        
        Args:
            input_grid: The input grid
            predicted_grid: The predicted output
            expected_grid: The expected output
            title: Overall title
        """
        _ensure_matplotlib()
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        self.show_grid(input_grid, "Input", ax=axes[0])
        self.show_grid(predicted_grid, "Predicted", ax=axes[1])
        self.show_grid(expected_grid, "Expected", ax=axes[2])
        
        # Show difference
        self._show_diff(predicted_grid, expected_grid, ax=axes[3])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _show_diff(self, predicted, expected, ax):
        """
        Show the difference between predicted and expected.
        
        Args:
            predicted: Predicted grid
            expected: Expected grid
            ax: Matplotlib axes
        """
        # Get data
        pred_data = predicted.data if hasattr(predicted, 'data') else np.array(predicted)
        exp_data = expected.data if hasattr(expected, 'data') else np.array(expected)
        
        # Handle shape mismatch
        if pred_data.shape != exp_data.shape:
            ax.text(0.5, 0.5, "Shape\nMismatch", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, color='red')
            ax.set_title("Difference")
            return
        
        # Create diff visualization
        diff = pred_data != exp_data
        
        # Use a red overlay for errors
        display = np.zeros((*exp_data.shape, 3))
        display[~diff] = [0.2, 0.8, 0.2]  # Green for correct
        display[diff] = [0.9, 0.2, 0.2]   # Red for errors
        
        ax.imshow(display)
        ax.set_title(f"Difference ({np.sum(diff)} errors)")
        ax.set_xticks([])
        ax.set_yticks([])
    
    def show_task(self, task, show_test: bool = True):
        """
        Display a complete ARC task with all examples.
        
        Args:
            task: ARCTask object
            show_test: Whether to show test inputs
        """
        _ensure_matplotlib()
        
        n_train = len(task.train_pairs)
        n_test = len(task.test_pairs) if show_test else 0
        n_total = n_train + n_test
        
        fig, axes = plt.subplots(n_total, 2, figsize=(8, 3 * n_total))
        
        if n_total == 1:
            axes = [axes]
        
        # Show training pairs
        for i, pair in enumerate(task.train_pairs):
            self.show_grid(pair.input_grid, f"Train {i+1} Input", ax=axes[i][0])
            self.show_grid(pair.output_grid, f"Train {i+1} Output", ax=axes[i][1])
        
        # Show test pairs
        if show_test:
            for i, pair in enumerate(task.test_pairs):
                idx = n_train + i
                self.show_grid(pair.input_grid, f"Test {i+1} Input", ax=axes[idx][0])
                if pair.output_grid:
                    self.show_grid(pair.output_grid, f"Test {i+1} Output", ax=axes[idx][1])
                else:
                    axes[idx][1].text(0.5, 0.5, "?", 
                                     transform=axes[idx][1].transAxes,
                                     ha='center', va='center', fontsize=48)
                    axes[idx][1].set_title("Test Output (Unknown)")
        
        fig.suptitle(f"Task: {task.task_id}", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def show_analysis_dashboard(
        self, 
        analysis_result,
        input_grid=None
    ):
        """
        Display a comprehensive analysis dashboard.
        
        Args:
            analysis_result: AnalysisResult object
            input_grid: Optional input grid to display
        """
        _ensure_matplotlib()
        
        n_cols = 4 if input_grid else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        
        idx = 0
        
        # Show input if provided
        if input_grid:
            self.show_grid(input_grid, "Input", ax=axes[idx])
            idx += 1
        
        # Show predicted
        if analysis_result.predicted_grid:
            self.show_grid(analysis_result.predicted_grid, "Predicted", ax=axes[idx])
        idx += 1
        
        # Show expected
        if analysis_result.expected_grid:
            self.show_grid(analysis_result.expected_grid, "Expected", ax=axes[idx])
        idx += 1
        
        # Show metrics/info
        ax = axes[idx]
        ax.axis('off')
        
        info_text = [
            f"Correct: {'✓' if analysis_result.is_correct else '✗'}",
            f"Accuracy: {analysis_result.accuracy:.1%}",
            "",
            "Metrics:"
        ]
        
        for key, value in list(analysis_result.metrics.items())[:5]:
            info_text.append(f"  {key}: {value:.3f}")
        
        ax.text(0.1, 0.9, "\n".join(info_text), 
               transform=ax.transAxes, va='top',
               fontsize=12, family='monospace')
        ax.set_title("Analysis")
        
        status = "✓ CORRECT" if analysis_result.is_correct else "✗ INCORRECT"
        color = "green" if analysis_result.is_correct else "red"
        fig.suptitle(status, fontsize=16, color=color)
        
        plt.tight_layout()
        plt.show()
    
    def save_figure(self, filename: str, dpi: int = None):
        """
        Save the current figure to a file.
        
        Args:
            filename: Output filename
            dpi: Dots per inch (optional)
        """
        _ensure_matplotlib()
        plt.savefig(filename, dpi=dpi or self.dpi, bbox_inches='tight')
        print(f"Figure saved to: {filename}")
    
    def show_color_legend(self):
        """Display the ARC color palette legend."""
        _ensure_matplotlib()
        
        fig, ax = plt.subplots(figsize=(10, 1))
        
        for i, color in enumerate(ARC_COLORS):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
            ax.text(i + 0.5, 0.5, str(i), ha='center', va='center', 
                   color='white' if i in [0, 9] else 'black', fontsize=12)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ARC Color Palette")
        
        plt.tight_layout()
        plt.show()

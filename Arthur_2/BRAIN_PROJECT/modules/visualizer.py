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
            fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
        
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
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
        
        self.show_grid(input_grid, "Input", ax=axes[0])
        self.show_grid(output_grid, "Output", ax=axes[1])
        
        fig.suptitle(title, fontsize=14)
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
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
        
        self.show_grid(input_grid, "Input", ax=axes[0])
        self.show_grid(predicted_grid, "Predicted", ax=axes[1])
        self.show_grid(expected_grid, "Expected", ax=axes[2])
        
        # Show difference
        self._show_diff(predicted_grid, expected_grid, ax=axes[3])
        
        fig.suptitle(title, fontsize=14)
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
        
        fig, axes = plt.subplots(n_total, 2, figsize=(8, 3 * n_total), layout='constrained')
        
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
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), layout='constrained')
        
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
        
        fig, ax = plt.subplots(figsize=(10, 1), layout='constrained')
        
        for i, color in enumerate(ARC_COLORS):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
            ax.text(i + 0.5, 0.5, str(i), ha='center', va='center', 
                   color='white' if i in [0, 9] else 'black', fontsize=12)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ARC Color Palette")
        
        plt.show()
    
    def create_batch_summary(
        self,
        task_visuals: List[Dict[str, Any]],
        title: str = "Batch Evaluation Results",
        save_path: Optional[str] = None,
        show: bool = True,
        max_cols: int = 4
    ):
        """
        Create a summary visualization of all batch results.
        
        Each task is shown as a row with: Input | Predicted | Expected | Status
        
        Args:
            task_visuals: List of dicts with keys:
                - task_id: str
                - input_grid: Grid or array
                - predicted_grid: Grid or array (can be None)
                - expected_grid: Grid or array (can be None)
                - is_correct: bool
                - accuracy: float
            title: Overall title
            save_path: Path to save the figure (optional)
            show: Whether to display the figure
            max_cols: Max tasks per row in grid layout
            
        Returns:
            The matplotlib figure object
        """
        _ensure_matplotlib()
        
        n_tasks = len(task_visuals)
        if n_tasks == 0:
            return None
        
        # Calculate grid dimensions
        # Each task needs 3 columns: Input, Predicted, Expected
        cols_per_task = 3
        
        # Determine layout
        if n_tasks <= 4:
            # Single row for few tasks
            n_rows = 1
            n_cols = n_tasks * cols_per_task
            fig_width = 3 * n_tasks
            fig_height = 4
        else:
            # Grid layout for many tasks
            tasks_per_row = min(max_cols, n_tasks)
            n_rows = (n_tasks + tasks_per_row - 1) // tasks_per_row
            n_cols = tasks_per_row * cols_per_task
            fig_width = 3 * tasks_per_row
            fig_height = 3.5 * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), layout='constrained')
        
        # Ensure axes is always 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Hide all axes initially
        for row in axes:
            for ax in row:
                ax.axis('off')
        
        # Plot each task
        for i, task_data in enumerate(task_visuals):
            if n_tasks <= 4:
                row_idx = 0
                col_offset = i * cols_per_task
            else:
                tasks_per_row = min(max_cols, n_tasks)
                row_idx = i // tasks_per_row
                col_offset = (i % tasks_per_row) * cols_per_task
            
            task_id = task_data.get('task_id', f'Task {i+1}')
            is_correct = task_data.get('is_correct', False)
            accuracy = task_data.get('accuracy', 0.0)
            
            # Status indicator
            status = "✓" if is_correct else f"✗ {accuracy:.0%}"
            status_color = "green" if is_correct else "red"
            
            # Input grid
            ax_input = axes[row_idx, col_offset]
            ax_input.axis('on')
            if task_data.get('input_grid') is not None:
                self.show_grid(task_data['input_grid'], "", ax=ax_input)
            ax_input.set_title(f"{task_id}\nInput", fontsize=8)
            
            # Predicted grid
            ax_pred = axes[row_idx, col_offset + 1]
            ax_pred.axis('on')
            if task_data.get('predicted_grid') is not None:
                self.show_grid(task_data['predicted_grid'], "", ax=ax_pred)
                ax_pred.set_title(f"Predicted\n{status}", fontsize=8, color=status_color)
            else:
                ax_pred.text(0.5, 0.5, "No\nPrediction", ha='center', va='center',
                           fontsize=10, color='gray', transform=ax_pred.transAxes)
                ax_pred.set_title("Predicted", fontsize=8)
            
            # Expected grid
            ax_exp = axes[row_idx, col_offset + 2]
            ax_exp.axis('on')
            if task_data.get('expected_grid') is not None:
                self.show_grid(task_data['expected_grid'], "", ax=ax_exp)
            ax_exp.set_title("Expected", fontsize=8)
        
        # Add overall title with summary stats
        n_correct = sum(1 for t in task_visuals if t.get('is_correct', False))
        avg_acc = sum(t.get('accuracy', 0) for t in task_visuals) / max(1, n_tasks)
        
        fig.suptitle(
            f"{title}\n{n_correct}/{n_tasks} correct ({n_correct/n_tasks:.0%}) | Avg accuracy: {avg_acc:.1%}",
            fontsize=12,
            fontweight='bold'
        )
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Batch summary saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def create_task_detail_image(
        self,
        task_id: str,
        input_grid,
        predicted_grid,
        expected_grid,
        is_correct: bool,
        accuracy: float,
        save_path: Optional[str] = None
    ):
        """
        Create a detailed comparison image for a single task.
        
        Args:
            task_id: Task identifier
            input_grid: Input grid
            predicted_grid: Predicted output (can be None)
            expected_grid: Expected output
            is_correct: Whether prediction was correct
            accuracy: Accuracy percentage
            save_path: Path to save the image
            
        Returns:
            The matplotlib figure object
        """
        _ensure_matplotlib()
        
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
        
        # Input
        self.show_grid(input_grid, "Input", ax=axes[0])
        
        # Predicted
        if predicted_grid is not None:
            self.show_grid(predicted_grid, "Predicted", ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "No Prediction", ha='center', va='center',
                        fontsize=12, color='gray', transform=axes[1].transAxes)
            axes[1].set_title("Predicted")
            axes[1].axis('off')
        
        # Expected
        self.show_grid(expected_grid, "Expected", ax=axes[2])
        
        # Difference
        if predicted_grid is not None:
            self._show_diff(predicted_grid, expected_grid, ax=axes[3])
        else:
            axes[3].axis('off')
        
        # Title with status
        status = "✓ CORRECT" if is_correct else f"✗ INCORRECT ({accuracy:.1%})"
        color = "green" if is_correct else "red"
        fig.suptitle(f"{task_id}: {status}", fontsize=14, color=color, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close(fig)
        
        return fig
    
    def create_interactive_browser(
        self,
        task_visuals: List[Dict[str, Any]],
        title: str = "Batch Results Browser"
    ):
        """
        Create an interactive browser to navigate through batch results.
        
        Uses matplotlib widgets for Previous/Next navigation.
        
        Args:
            task_visuals: List of dicts with keys:
                - task_id: str
                - input_grid: Grid or array
                - predicted_grid: Grid or array (can be None)
                - expected_grid: Grid or array (can be None)
                - is_correct: bool
                - accuracy: float
            title: Window title
        """
        _ensure_matplotlib()
        from matplotlib.widgets import Button
        
        if not task_visuals:
            print("No tasks to display")
            return
        
        # State
        current_idx = [0]  # Use list to allow modification in nested function
        n_tasks = len(task_visuals)
        
        # Calculate stats
        n_correct = sum(1 for t in task_visuals if t.get('is_correct', False))
        avg_acc = sum(t.get('accuracy', 0) for t in task_visuals) / max(1, n_tasks)
        
        # Create figure with space for buttons
        fig = plt.figure(figsize=(16, 6))
        
        # Create grid spec for layout
        gs = fig.add_gridspec(2, 4, height_ratios=[10, 1], hspace=0.3)
        
        # Axes for grids
        ax_input = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_exp = fig.add_subplot(gs[0, 2])
        ax_diff = fig.add_subplot(gs[0, 3])
        
        axes = [ax_input, ax_pred, ax_exp, ax_diff]
        
        # Button axes
        ax_prev = fig.add_subplot(gs[1, 0])
        ax_info = fig.add_subplot(gs[1, 1:3])
        ax_next = fig.add_subplot(gs[1, 3])
        
        # Info text area
        ax_info.axis('off')
        info_text = ax_info.text(
            0.5, 0.5, "", 
            ha='center', va='center',
            fontsize=12,
            transform=ax_info.transAxes
        )
        
        def update_display():
            """Update the display with current task."""
            idx = current_idx[0]
            task = task_visuals[idx]
            
            # Clear all axes
            for ax in axes:
                ax.clear()
            
            # Get data
            task_id = task.get('task_id', f'Task {idx+1}')
            is_correct = task.get('is_correct', False)
            accuracy = task.get('accuracy', 0.0)
            
            # Input grid
            if task.get('input_grid') is not None:
                self.show_grid(task['input_grid'], "Input", ax=ax_input)
            else:
                ax_input.text(0.5, 0.5, "No Data", ha='center', va='center',
                            fontsize=12, transform=ax_input.transAxes)
                ax_input.set_title("Input")
            
            # Predicted grid
            if task.get('predicted_grid') is not None:
                self.show_grid(task['predicted_grid'], "Predicted", ax=ax_pred)
            else:
                ax_pred.text(0.5, 0.5, "No\nPrediction", ha='center', va='center',
                           fontsize=12, color='gray', transform=ax_pred.transAxes)
                ax_pred.set_title("Predicted")
                ax_pred.axis('off')
            
            # Expected grid
            if task.get('expected_grid') is not None:
                self.show_grid(task['expected_grid'], "Expected", ax=ax_exp)
            else:
                ax_exp.text(0.5, 0.5, "No Data", ha='center', va='center',
                          fontsize=12, transform=ax_exp.transAxes)
                ax_exp.set_title("Expected")
            
            # Difference
            if task.get('predicted_grid') is not None and task.get('expected_grid') is not None:
                self._show_diff(task['predicted_grid'], task['expected_grid'], ax=ax_diff)
            else:
                ax_diff.axis('off')
                ax_diff.set_title("Difference")
            
            # Update title
            status = "✓ CORRECT" if is_correct else f"✗ INCORRECT ({accuracy:.1%})"
            status_color = "green" if is_correct else "red"
            fig.suptitle(
                f"{task_id}: {status}",
                fontsize=14, 
                color=status_color,
                fontweight='bold'
            )
            
            # Update info text
            info_text.set_text(
                f"Task {idx+1} / {n_tasks}  |  "
                f"Overall: {n_correct}/{n_tasks} correct ({n_correct/n_tasks:.0%})  |  "
                f"Avg accuracy: {avg_acc:.1%}"
            )
            
            fig.canvas.draw_idle()
        
        def on_prev(event):
            """Go to previous task."""
            current_idx[0] = (current_idx[0] - 1) % n_tasks
            update_display()
        
        def on_next(event):
            """Go to next task."""
            current_idx[0] = (current_idx[0] + 1) % n_tasks
            update_display()
        
        # Create buttons
        btn_prev = Button(ax_prev, '◀ Previous', color='lightgray', hovercolor='lightblue')
        btn_prev.on_clicked(on_prev)
        
        btn_next = Button(ax_next, 'Next ▶', color='lightgray', hovercolor='lightblue')
        btn_next.on_clicked(on_next)
        
        # Add keyboard navigation
        def on_key(event):
            if event.key == 'left':
                on_prev(event)
            elif event.key == 'right':
                on_next(event)
            elif event.key == 'escape' or event.key == 'q':
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Initial display
        update_display()
        
        # Window title
        fig.canvas.manager.set_window_title(title)
        
        plt.show()

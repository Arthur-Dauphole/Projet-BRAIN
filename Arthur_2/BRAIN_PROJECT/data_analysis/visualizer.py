"""
visualizer.py - Visualisations pour l'analyse de données
=========================================================

Génère des graphiques pour l'analyse des résultats de batch.
Optimisé pour les publications scientifiques IEEE (LaTeX/Overleaf compatible).

Output format: PDF vectoriel pour qualité publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


# =============================================================================
# PUBLICATION QUALITY CONFIGURATION (IEEE Standard)
# =============================================================================

def _setup_publication_style():
    """
    Configure matplotlib pour des figures de qualité publication IEEE.
    Détecte automatiquement si LaTeX est disponible sur le système.
    """
    # Check if LaTeX is available to avoid crashes
    latex_available = shutil.which('latex') is not None
    
    # IEEE column width: 3.5 inches, double column: 7.16 inches
    # Recommended figure sizes for IEEE publications
    IEEE_SINGLE_COLUMN = (3.5, 2.5)
    IEEE_DOUBLE_COLUMN = (7.16, 3.5)
    
    config = {
        # LaTeX rendering (only if available)
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if latex_available else ["DejaVu Serif"],
        
        # Font sizes (IEEE standard)
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        
        # Figure settings
        "figure.figsize": (6, 3.5),  # Default: between single and double column
        "figure.dpi": 150,           # Screen display
        "savefig.dpi": 300,          # High-res for publication
        "savefig.bbox": "tight",
        "savefig.format": "pdf",     # VECTORIAL OUTPUT IS MANDATORY
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
    }
    
    plt.rcParams.update(config)
    
    return latex_available


# Initialize publication style on module import
LATEX_AVAILABLE = _setup_publication_style()


class AnalysisVisualizer:
    """
    Crée des visualisations de qualité publication pour l'analyse des données.
    
    Features:
        - Compatible LaTeX/Overleaf (détection automatique)
        - Sortie vectorielle PDF par défaut
        - Dimensions IEEE (single/double column)
        - Palette de couleurs accessible (colorblind-friendly)
    
    Example:
        viz = AnalysisVisualizer(df)
        viz.plot_accuracy_by_transformation(save_path="figures/accuracy.pdf")
        viz.save_all("figures/", format="pdf")
    """
    
    # IEEE figure sizes
    IEEE_SINGLE_COLUMN = (3.5, 2.5)
    IEEE_DOUBLE_COLUMN = (7.16, 3.5)
    IEEE_FULL_PAGE = (7.16, 9.0)
    
    # Style for backward compatibility (will be overridden by global config)
    STYLE_CONFIG = {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (6, 3.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
    }
    
    # Palette de couleurs (Colorblind-friendly - Wong palette + extensions)
    # Based on: https://www.nature.com/articles/nmeth.1618
    COLORS = {
        "primary": "#0072B2",     # Blue (colorblind safe)
        "secondary": "#009E73",   # Green (colorblind safe)
        "accent": "#E69F00",      # Orange (colorblind safe)
        "error": "#D55E00",       # Vermillion (colorblind safe)
        "neutral": "#999999",     # Grey
        "success": "#009E73",     # Green
        "purple": "#CC79A7",      # Reddish purple (colorblind safe)
        "yellow": "#F0E442",      # Yellow (colorblind safe)
        "black": "#000000",       # Black
    }
    
    # Palette pour les transformations (colorblind-friendly)
    TRANSFORM_COLORS = {
        "translation": "#0072B2",   # Blue
        "rotation": "#D55E00",      # Vermillion
        "reflection": "#009E73",    # Green
        "color_change": "#E69F00",  # Orange
        "scale": "#CC79A7",         # Reddish purple
        "draw_line": "#56B4E9",     # Sky blue
        "add_border": "#F0E442",    # Yellow
        "composite": "#0072B2",     # Blue (darker)
        "tiling": "#009E73",        # Green
        "identity": "#999999",      # Grey
    }
    
    # Hatching patterns for additional distinction (accessibility)
    HATCH_PATTERNS = ['', '///', '...', 'xxx', '\\\\\\', 'ooo', '+++', '***']
    
    def __init__(self, df: pd.DataFrame, style: str = "publication"):
        """
        Initialise le visualiseur.
        
        Args:
            df: DataFrame avec les données
            style: Style de visualisation:
                   - "publication": IEEE/LaTeX optimized (PDF vectoriel)
                   - "presentation": Larger fonts for slides
                   - "default": matplotlib defaults
        """
        self.df = df
        self.figures: Dict[str, plt.Figure] = {}
        self.style = style
        self.latex_available = LATEX_AVAILABLE
        
        if style == "publication":
            _setup_publication_style()
        elif style == "presentation":
            plt.rcParams.update({
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 18,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "figure.figsize": (10, 6),
            })
    
    def _get_transform_color(self, transform: str) -> str:
        """Retourne la couleur pour un type de transformation."""
        return self.TRANSFORM_COLORS.get(transform, self.COLORS["neutral"])
    
    def _get_figsize(self, size: str = "double") -> Tuple[float, float]:
        """
        Retourne la taille IEEE appropriée.
        
        Args:
            size: "single" (3.5in), "double" (7.16in), "full" (full page)
        """
        sizes = {
            "single": self.IEEE_SINGLE_COLUMN,
            "double": self.IEEE_DOUBLE_COLUMN,
            "full": self.IEEE_FULL_PAGE,
        }
        return sizes.get(size, self.IEEE_DOUBLE_COLUMN)
    
    def _save_figure(self, fig: plt.Figure, save_path: str, formats: List[str] = None):
        """
        Sauvegarde la figure dans plusieurs formats.
        
        Args:
            fig: Figure matplotlib
            save_path: Chemin de base (sans extension)
            formats: Liste de formats (défaut: ["pdf", "png"])
        """
        if formats is None:
            formats = ["pdf"]  # PDF vectoriel par défaut pour IEEE
        
        base_path = Path(save_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove extension if present
        if base_path.suffix:
            base_path = base_path.with_suffix("")
        
        for fmt in formats:
            filepath = base_path.with_suffix(f".{fmt}")
            fig.savefig(filepath, format=fmt, dpi=300 if fmt == "png" else None)
            print(f"Saved: {filepath}")
    
    # ==================== BAR PLOTS ====================
    
    def plot_accuracy_by_transformation(
        self, 
        figsize: Tuple[float, float] = None,
        ieee_size: str = "double",
        show_std: bool = True,
        show_n: bool = True,
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Barplot de l'accuracy par type de transformation.
        
        Args:
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            show_std: Afficher les barres d'erreur
            show_n: Afficher le nombre d'échantillons
            save_path: Chemin pour sauvegarder (sans extension pour multi-format)
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        if "primary_transformation" not in self.df.columns:
            print("Column 'primary_transformation' not found")
            return None
        
        # Calculer les statistiques
        stats = self.df.groupby("primary_transformation")["accuracy"].agg(["mean", "std", "count"])
        stats = stats.sort_values("mean", ascending=True)
        
        # Taille de la figure
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Couleurs pour chaque barre
        colors = [self._get_transform_color(t) for t in stats.index]
        
        # Barplot horizontal
        bars = ax.barh(
            stats.index, 
            stats["mean"], 
            xerr=stats["std"] if show_std else None,
            color=colors,
            capsize=2,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8}
        )
        
        # Ajouter les valeurs sur les barres
        for bar, (idx, row) in zip(bars, stats.iterrows()):
            width = bar.get_width()
            label = f'{width:.1%}'
            if show_n:
                label += f' ($n$={int(row["count"])})'
            ax.text(
                min(width + 0.02, 1.0), 
                bar.get_y() + bar.get_height()/2,
                label,
                va='center',
                fontsize=7,
                color='black'
            )
        
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Transformation Type")
        ax.set_xlim(0, 1.12)
        ax.axvline(x=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["accuracy_by_transformation"] = fig
        return fig
    
    def plot_model_comparison(
        self, 
        figsize: Tuple[float, float] = None,
        ieee_size: str = "single",
        metric: str = "accuracy",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Compare les performances de différents modèles.
        
        Args:
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            metric: Métrique à comparer
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        if "model" not in self.df.columns:
            print("Column 'model' not found")
            return None
        
        # Calculer les statistiques
        stats = self.df.groupby("model")[metric].agg(["mean", "std", "count"])
        stats = stats.sort_values("mean", ascending=False)
        
        # Taille de la figure
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(stats))
        bars = ax.bar(
            x, 
            stats["mean"],
            yerr=stats["std"],
            color=self.COLORS["primary"],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8}
        )
        
        # Ajouter les valeurs
        for bar, (idx, row) in zip(bars, stats.iterrows()):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + row["std"] + 0.02,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index, rotation=45, ha='right')
        ax.set_ylabel(f"Mean {metric.replace('_', ' ').title()}")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["model_comparison"] = fig
        return fig
    
    # ==================== BOX PLOTS ====================
    
    def plot_accuracy_boxplot(
        self,
        group_by: str = "primary_transformation",
        figsize: Tuple[float, float] = None,
        ieee_size: str = "double",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Boxplot de l'accuracy groupée.
        
        Args:
            group_by: Colonne de groupement
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        if group_by not in self.df.columns:
            print(f"Column '{group_by}' not found")
            return None
        
        # Taille de la figure
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Préparer les données
        groups = self.df.groupby(group_by)["accuracy"].apply(list).to_dict()
        sorted_groups = sorted(groups.items(), key=lambda x: np.mean(x[1]), reverse=True)
        
        labels = [g[0] for g in sorted_groups]
        data = [g[1] for g in sorted_groups]
        colors = [self._get_transform_color(l) for l in labels]
        
        # Boxplot
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=False,  # notch=False for cleaner look
            widths=0.6,
            flierprops=dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='black'),
            medianprops=dict(color='black', linewidth=1.2),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            boxprops=dict(linewidth=0.8)
        )
        
        # Colorer les boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        ax.set_ylabel("Accuracy")
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.axhline(y=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_ylim(-0.05, 1.1)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["accuracy_boxplot"] = fig
        return fig
    
    # ==================== HEATMAPS ====================
    
    def plot_confusion_matrix(
        self,
        figsize: Tuple[float, float] = None,
        ieee_size: str = "single",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Matrice de confusion: transformation détectée vs action utilisée.
        
        Args:
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        if "primary_transformation" not in self.df.columns or "action_used" not in self.df.columns:
            print("Required columns not found")
            return None
        
        # Créer la matrice de confusion
        confusion = pd.crosstab(
            self.df["primary_transformation"],
            self.df["action_used"],
            margins=True
        )
        
        # Taille de la figure (square for matrix)
        if figsize is None:
            base = self._get_figsize(ieee_size)
            figsize = (base[0], base[0])  # Square
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Heatmap with sequential colormap
        im = ax.imshow(confusion.iloc[:-1, :-1], cmap="Blues", aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(confusion.columns) - 1))
        ax.set_yticks(range(len(confusion.index) - 1))
        ax.set_xticklabels(confusion.columns[:-1], rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(confusion.index[:-1], fontsize=7)
        
        # Ajouter les valeurs
        max_val = confusion.iloc[:-1, :-1].values.max()
        for i in range(len(confusion.index) - 1):
            for j in range(len(confusion.columns) - 1):
                value = confusion.iloc[i, j]
                ax.text(j, i, str(value), ha='center', va='center', 
                       color='white' if value > max_val / 2 else 'black',
                       fontsize=7)
        
        ax.set_xlabel("Action Used")
        ax.set_ylabel("Detected Transformation")
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Count', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["confusion_matrix"] = fig
        return fig
    
    # ==================== LINE PLOTS ====================
    
    def plot_timing_breakdown(
        self,
        figsize: Tuple[float, float] = None,
        ieee_size: str = "double",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Stacked bar chart du temps par composante.
        
        Args:
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        timing_cols = ["timing_detection", "timing_llm_response", "timing_action_execution"]
        available_cols = [c for c in timing_cols if c in self.df.columns]
        
        if not available_cols:
            print("No timing columns found")
            return None
        
        # Taille de la figure
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Grouper par transformation si possible
        if "primary_transformation" in self.df.columns:
            means = self.df.groupby("primary_transformation")[available_cols].mean()
            means = means.sort_values(available_cols[0], ascending=False)
        else:
            means = self.df[available_cols].mean().to_frame().T
            means.index = ["All"]
        
        # Stacked bar with colorblind-safe colors
        colors = [self.COLORS["secondary"], self.COLORS["primary"], self.COLORS["accent"]]
        labels = ["Detection", "LLM Response", "Execution"]
        hatches = ['', '///', '...']  # Additional visual distinction
        
        x = range(len(means))
        bottom = np.zeros(len(means))
        
        for i, (col, color, label, hatch) in enumerate(zip(available_cols, colors, labels, hatches)):
            if col in means.columns:
                bars = ax.bar(x, means[col], bottom=bottom, label=label, 
                             color=color, edgecolor='black', linewidth=0.5, hatch=hatch)
                bottom += means[col].values
        
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Time (seconds)")
        ax.legend(loc='upper right', fontsize=7)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["timing_breakdown"] = fig
        return fig
    
    def plot_llm_vs_fallback(
        self,
        figsize: Tuple[float, float] = None,
        ieee_size: str = "double",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Compare les performances LLM vs Fallback.
        
        Args:
            figsize: Taille de la figure (override ieee_size)
            ieee_size: Taille IEEE ("single", "double", "full")
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie (défaut: ["pdf"])
            
        Returns:
            Figure matplotlib
        """
        if "was_fallback_used" not in self.df.columns:
            print("Column 'was_fallback_used' not found")
            return None
        
        # Taille de la figure
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart: proportion LLM vs Fallback
        fallback_counts = self.df["was_fallback_used"].value_counts()
        labels = ["LLM Only", "Fallback"]
        colors = [self.COLORS["primary"], self.COLORS["accent"]]
        
        # Pie chart with clean styling
        wedges, texts, autotexts = axes[0].pie(
            fallback_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=0.5),
            textprops=dict(fontsize=8)
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        # Bar chart: accuracy comparison
        llm_acc = self.df[self.df["was_fallback_used"] == False]["accuracy"].mean()
        fallback_acc = self.df[self.df["was_fallback_used"] == True]["accuracy"].mean()
        
        # Handle NaN values
        llm_acc = llm_acc if not np.isnan(llm_acc) else 0
        fallback_acc = fallback_acc if not np.isnan(fallback_acc) else 0
        
        bars = axes[1].bar(
            ["LLM Only", "Fallback"],
            [llm_acc, fallback_acc],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        axes[1].set_ylabel("Mean Accuracy")
        axes[1].set_ylim(0, 1.1)
        axes[1].axhline(y=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Clean up spines
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["llm_vs_fallback"] = fig
        return fig
    
    # ==================== ADDITIONAL PLOTS FOR IEEE ====================
    
    def plot_accuracy_by_complexity(
        self,
        complexity_col: str = "complexity_num_objects",
        figsize: Tuple[float, float] = None,
        ieee_size: str = "single",
        save_path: Optional[str] = None,
        save_formats: List[str] = None
    ) -> plt.Figure:
        """
        Scatter plot accuracy vs complexity.
        
        Args:
            complexity_col: Colonne de complexité
            figsize: Taille de la figure
            ieee_size: Taille IEEE
            save_path: Chemin pour sauvegarder
            save_formats: Formats de sortie
            
        Returns:
            Figure matplotlib
        """
        if complexity_col not in self.df.columns or "accuracy" not in self.df.columns:
            print(f"Required columns not found: {complexity_col}")
            return None
        
        # Filter valid data
        valid_data = self.df[[complexity_col, "accuracy"]].dropna()
        
        if len(valid_data) < 3:
            print(f"Not enough valid data points for complexity plot")
            return None
        
        # Check if complexity values have variance
        if valid_data[complexity_col].std() == 0:
            print(f"No variance in {complexity_col} - cannot plot trend")
            return None
        
        if figsize is None:
            figsize = self._get_figsize(ieee_size)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(
            valid_data[complexity_col],
            valid_data["accuracy"],
            alpha=0.6,
            color=self.COLORS["primary"],
            edgecolor='black',
            linewidth=0.3,
            s=30
        )
        
        # Trend line (with error handling)
        try:
            x_vals = valid_data[complexity_col].values.astype(float)
            y_vals = valid_data["accuracy"].values.astype(float)
            
            # Check for valid numerical data
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_clean = x_vals[mask]
            y_clean = y_vals[mask]
            
            if len(x_clean) > 2 and np.std(x_clean) > 0:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax.plot(x_line, p(x_line), color=self.COLORS["error"], 
                       linestyle='--', linewidth=1.2, label='Trend')
                ax.legend(fontsize=7)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Skip trend line if regression fails
            print(f"Warning: Could not compute trend line: {e}")
        
        ax.set_xlabel(complexity_col.replace('_', ' ').title())
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color=self.COLORS["neutral"], linestyle=':', alpha=0.5, linewidth=0.8)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, save_formats)
        
        self.figures["accuracy_by_complexity"] = fig
        return fig
    
    # ==================== SAVE ALL ====================
    
    def save_all(
        self, 
        output_dir: str = "figures/",
        formats: List[str] = None
    ):
        """
        Sauvegarde toutes les figures générées.
        
        Args:
            output_dir: Répertoire de sortie
            formats: Formats d'image (défaut: ["pdf", "png"] pour IEEE)
        """
        if formats is None:
            formats = ["pdf", "png"]  # PDF vectoriel + PNG raster
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            for fmt in formats:
                filepath = output_path / f"{name}.{fmt}"
                fig.savefig(filepath, format=fmt, dpi=300 if fmt == "png" else None)
                print(f"Saved: {filepath}")
    
    def show_all(self):
        """Affiche toutes les figures."""
        plt.show()
    
    def generate_all_plots(
        self,
        output_dir: str = "figures/",
        formats: List[str] = None
    ):
        """
        Génère et sauvegarde toutes les visualisations disponibles.
        
        Args:
            output_dir: Répertoire de sortie
            formats: Formats de sortie (défaut: ["pdf"])
        """
        if formats is None:
            formats = ["pdf"]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all available plots
        print("Generating publication-quality figures...")
        
        self.plot_accuracy_by_transformation(
            save_path=str(output_path / "accuracy_by_transformation"),
            save_formats=formats
        )
        
        self.plot_accuracy_boxplot(
            save_path=str(output_path / "accuracy_boxplot"),
            save_formats=formats
        )
        
        self.plot_timing_breakdown(
            save_path=str(output_path / "timing_breakdown"),
            save_formats=formats
        )
        
        self.plot_llm_vs_fallback(
            save_path=str(output_path / "llm_vs_fallback"),
            save_formats=formats
        )
        
        # Optional plots (may fail if columns don't exist)
        try:
            self.plot_confusion_matrix(
                save_path=str(output_path / "confusion_matrix"),
                save_formats=formats
            )
        except Exception:
            pass
        
        try:
            self.plot_model_comparison(
                save_path=str(output_path / "model_comparison"),
                save_formats=formats
            )
        except Exception:
            pass
        
        try:
            self.plot_accuracy_by_complexity(
                save_path=str(output_path / "accuracy_by_complexity"),
                save_formats=formats
            )
        except Exception:
            pass
        
        print(f"\nAll figures saved to: {output_path}")

"""
visualizer.py - Visualisations pour l'analyse de données
=========================================================

Génère des graphiques pour l'analyse des résultats de batch.
Optimisé pour les publications scientifiques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


class AnalysisVisualizer:
    """
    Crée des visualisations pour l'analyse des données.
    
    Example:
        viz = AnalysisVisualizer(df)
        viz.plot_accuracy_by_transformation()
        viz.plot_model_comparison()
        viz.save_all("figures/")
    """
    
    # Style pour publications scientifiques
    STYLE_CONFIG = {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    
    # Palette de couleurs
    COLORS = {
        "primary": "#2563eb",     # Bleu
        "secondary": "#10b981",   # Vert
        "accent": "#f59e0b",      # Orange
        "error": "#ef4444",       # Rouge
        "neutral": "#6b7280",     # Gris
        "success": "#22c55e",     # Vert succès
    }
    
    # Palette pour les transformations
    TRANSFORM_COLORS = {
        "translation": "#3b82f6",
        "rotation": "#ef4444",
        "reflection": "#22c55e",
        "color_change": "#f59e0b",
        "scale": "#8b5cf6",
        "draw_line": "#ec4899",
        "add_border": "#06b6d4",
        "composite": "#6366f1",
        "tiling": "#14b8a6",
        "identity": "#9ca3af",
    }
    
    def __init__(self, df: pd.DataFrame, style: str = "publication"):
        """
        Initialise le visualiseur.
        
        Args:
            df: DataFrame avec les données
            style: Style de visualisation ("publication", "presentation", "default")
        """
        self.df = df
        self.figures: Dict[str, plt.Figure] = {}
        
        if style == "publication":
            plt.rcParams.update(self.STYLE_CONFIG)
    
    def _get_transform_color(self, transform: str) -> str:
        """Retourne la couleur pour un type de transformation."""
        return self.TRANSFORM_COLORS.get(transform, self.COLORS["neutral"])
    
    # ==================== BAR PLOTS ====================
    
    def plot_accuracy_by_transformation(
        self, 
        figsize: Tuple[int, int] = (10, 6),
        show_std: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Barplot de l'accuracy par type de transformation.
        
        Args:
            figsize: Taille de la figure
            show_std: Afficher les barres d'erreur
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        if "primary_transformation" not in self.df.columns:
            print("Column 'primary_transformation' not found")
            return None
        
        # Calculer les statistiques
        stats = self.df.groupby("primary_transformation")["accuracy"].agg(["mean", "std", "count"])
        stats = stats.sort_values("mean", ascending=True)
        
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
            capsize=3,
            edgecolor="white",
            linewidth=0.5
        )
        
        # Ajouter les valeurs sur les barres
        for bar, (idx, row) in zip(bars, stats.iterrows()):
            width = bar.get_width()
            ax.text(
                width + 0.02, 
                bar.get_y() + bar.get_height()/2,
                f'{width:.1%} (n={int(row["count"])})',
                va='center',
                fontsize=9
            )
        
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy by Transformation Type")
        ax.set_xlim(0, 1.15)
        ax.axvline(x=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["accuracy_by_transformation"] = fig
        return fig
    
    def plot_model_comparison(
        self, 
        figsize: Tuple[int, int] = (10, 6),
        metric: str = "accuracy",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare les performances de différents modèles.
        
        Args:
            figsize: Taille de la figure
            metric: Métrique à comparer
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        if "model" not in self.df.columns:
            print("Column 'model' not found")
            return None
        
        # Calculer les statistiques
        stats = self.df.groupby("model")[metric].agg(["mean", "std", "count"])
        stats = stats.sort_values("mean", ascending=False)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(stats))
        bars = ax.bar(
            x, 
            stats["mean"],
            yerr=stats["std"],
            color=self.COLORS["primary"],
            capsize=5,
            edgecolor="white",
            linewidth=0.5
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
                fontsize=10
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index, rotation=45, ha='right')
        ax.set_ylabel(f"Mean {metric.title()}")
        ax.set_title(f"Model Comparison: {metric.title()}")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["model_comparison"] = fig
        return fig
    
    # ==================== BOX PLOTS ====================
    
    def plot_accuracy_boxplot(
        self,
        group_by: str = "primary_transformation",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Boxplot de l'accuracy groupée.
        
        Args:
            group_by: Colonne de groupement
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        if group_by not in self.df.columns:
            print(f"Column '{group_by}' not found")
            return None
        
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
            notch=True
        )
        
        # Colorer les boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy Distribution by {group_by.replace('_', ' ').title()}")
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(y=1.0, color=self.COLORS["neutral"], linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["accuracy_boxplot"] = fig
        return fig
    
    # ==================== HEATMAPS ====================
    
    def plot_confusion_matrix(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Matrice de confusion: transformation détectée vs action utilisée.
        
        Args:
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder
            
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
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Heatmap
        im = ax.imshow(confusion.iloc[:-1, :-1], cmap="Blues", aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(confusion.columns) - 1))
        ax.set_yticks(range(len(confusion.index) - 1))
        ax.set_xticklabels(confusion.columns[:-1], rotation=45, ha='right')
        ax.set_yticklabels(confusion.index[:-1])
        
        # Ajouter les valeurs
        for i in range(len(confusion.index) - 1):
            for j in range(len(confusion.columns) - 1):
                value = confusion.iloc[i, j]
                ax.text(j, i, str(value), ha='center', va='center', 
                       color='white' if value > confusion.values.max() / 2 else 'black')
        
        ax.set_xlabel("Action Used")
        ax.set_ylabel("Detected Transformation")
        ax.set_title("Confusion Matrix: Detection vs Execution")
        
        plt.colorbar(im, ax=ax, label='Count')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["confusion_matrix"] = fig
        return fig
    
    # ==================== LINE PLOTS ====================
    
    def plot_timing_breakdown(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Stacked bar chart du temps par composante.
        
        Args:
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        timing_cols = ["timing_detection", "timing_llm_response", "timing_action_execution"]
        available_cols = [c for c in timing_cols if c in self.df.columns]
        
        if not available_cols:
            print("No timing columns found")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Grouper par transformation si possible
        if "primary_transformation" in self.df.columns:
            means = self.df.groupby("primary_transformation")[available_cols].mean()
            means = means.sort_values(available_cols[0], ascending=False)
        else:
            means = self.df[available_cols].mean().to_frame().T
            means.index = ["All"]
        
        # Stacked bar
        colors = [self.COLORS["secondary"], self.COLORS["primary"], self.COLORS["accent"]]
        labels = ["Detection", "LLM Response", "Action Execution"]
        
        x = range(len(means))
        bottom = np.zeros(len(means))
        
        for i, (col, color, label) in enumerate(zip(available_cols, colors, labels)):
            if col in means.columns:
                ax.bar(x, means[col], bottom=bottom, label=label, color=color)
                bottom += means[col].values
        
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Timing Breakdown by Transformation")
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["timing_breakdown"] = fig
        return fig
    
    def plot_llm_vs_fallback(
        self,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare les performances LLM vs Fallback.
        
        Args:
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        if "was_fallback_used" not in self.df.columns:
            print("Column 'was_fallback_used' not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart: proportion LLM vs Fallback
        fallback_counts = self.df["was_fallback_used"].value_counts()
        labels = ["LLM Only", "With Fallback"]
        colors = [self.COLORS["primary"], self.COLORS["accent"]]
        
        axes[0].pie(
            fallback_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        axes[0].set_title("LLM vs Fallback Usage")
        
        # Bar chart: accuracy comparison
        llm_acc = self.df[self.df["was_fallback_used"] == False]["accuracy"].mean()
        fallback_acc = self.df[self.df["was_fallback_used"] == True]["accuracy"].mean()
        
        bars = axes[1].bar(
            ["LLM Only", "With Fallback"],
            [llm_acc, fallback_acc],
            color=colors
        )
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                height + 0.02,
                f'{height:.1%}',
                ha='center',
                va='bottom'
            )
        
        axes[1].set_ylabel("Mean Accuracy")
        axes[1].set_title("Accuracy Comparison")
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            print(f"Saved: {save_path}")
        
        self.figures["llm_vs_fallback"] = fig
        return fig
    
    # ==================== SAVE ALL ====================
    
    def save_all(
        self, 
        output_dir: str = "figures/",
        format: str = "png"
    ):
        """
        Sauvegarde toutes les figures générées.
        
        Args:
            output_dir: Répertoire de sortie
            format: Format d'image (png, pdf, svg)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = output_path / f"{name}.{format}"
            fig.savefig(filepath)
            print(f"Saved: {filepath}")
    
    def show_all(self):
        """Affiche toutes les figures."""
        plt.show()

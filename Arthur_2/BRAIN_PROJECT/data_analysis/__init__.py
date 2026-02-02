"""
data_analysis - Module d'analyse de données pour BRAIN Project
===============================================================

Ce module fournit des outils pour analyser les résultats des batchs
et générer des visualisations pour articles scientifiques.

Composants:
    - DataLoader: Charger et agréger les résultats de plusieurs batchs
    - Metrics: Calculer des métriques statistiques
    - AnalysisVisualizer: Créer des graphiques (barplots, heatmaps, etc.)
    - ReportGenerator: Exporter vers LaTeX/CSV

Usage:
    from data_analysis import DataLoader, AnalysisVisualizer
    
    # Charger les données
    loader = DataLoader()
    df = loader.load_all_batches("results/")
    
    # Créer des visualisations
    viz = AnalysisVisualizer(df)
    viz.plot_accuracy_by_transformation()
    viz.plot_model_comparison()
"""

from .data_loader import DataLoader
from .metrics import MetricsCalculator
from .visualizer import AnalysisVisualizer
from .report_generator import ReportGenerator

__all__ = [
    'DataLoader',
    'MetricsCalculator', 
    'AnalysisVisualizer',
    'ReportGenerator'
]

__version__ = "1.0.0"

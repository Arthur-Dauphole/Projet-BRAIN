"""
metrics.py - Calcul de métriques statistiques
==============================================

Fournit des fonctions pour calculer diverses métriques
à partir des données de batch.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats


class MetricsCalculator:
    """
    Calcule des métriques statistiques sur les résultats de batch.
    
    Example:
        calc = MetricsCalculator(df)
        print(calc.overall_accuracy())
        print(calc.accuracy_by_transformation())
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise le calculateur avec un DataFrame.
        
        Args:
            df: DataFrame avec les résultats des tâches
        """
        self.df = df
    
    # ==================== ACCURACY METRICS ====================
    
    def overall_accuracy(self) -> Dict[str, float]:
        """
        Calcule l'accuracy globale.
        
        Returns:
            Dict avec mean, std, min, max, median
        """
        if "accuracy" not in self.df.columns:
            return {}
        
        acc = self.df["accuracy"]
        return {
            "mean": acc.mean(),
            "std": acc.std(),
            "min": acc.min(),
            "max": acc.max(),
            "median": acc.median(),
            "count": len(acc)
        }
    
    def accuracy_by_transformation(self) -> pd.DataFrame:
        """
        Calcule l'accuracy par type de transformation.
        
        Returns:
            DataFrame avec mean, std, count par transformation
        """
        if "primary_transformation" not in self.df.columns:
            return pd.DataFrame()
        
        grouped = self.df.groupby("primary_transformation")["accuracy"].agg([
            "mean", "std", "count", "min", "max"
        ]).round(4)
        
        # Ajouter le taux de succès (100% accuracy)
        success_rate = self.df.groupby("primary_transformation")["is_correct"].mean()
        grouped["success_rate"] = success_rate.round(4)
        
        return grouped.sort_values("mean", ascending=False)
    
    def accuracy_by_model(self) -> pd.DataFrame:
        """
        Calcule l'accuracy par modèle LLM.
        
        Returns:
            DataFrame avec mean, std, count par modèle
        """
        if "model" not in self.df.columns:
            return pd.DataFrame()
        
        grouped = self.df.groupby("model")["accuracy"].agg([
            "mean", "std", "count", "min", "max"
        ]).round(4)
        
        success_rate = self.df.groupby("model")["is_correct"].mean()
        grouped["success_rate"] = success_rate.round(4)
        
        return grouped.sort_values("mean", ascending=False)
    
    def accuracy_by_complexity(
        self, 
        complexity_col: str = "complexity_num_colors"
    ) -> pd.DataFrame:
        """
        Calcule l'accuracy en fonction de la complexité.
        
        Args:
            complexity_col: Colonne de complexité à utiliser
            
        Returns:
            DataFrame groupé par niveau de complexité
        """
        if complexity_col not in self.df.columns:
            return pd.DataFrame()
        
        grouped = self.df.groupby(complexity_col)["accuracy"].agg([
            "mean", "std", "count"
        ]).round(4)
        
        return grouped
    
    # ==================== LLM vs FALLBACK ====================
    
    def llm_vs_fallback_comparison(self) -> Dict[str, Any]:
        """
        Compare les performances LLM seul vs avec fallback.
        
        Returns:
            Dict avec les métriques de comparaison
        """
        if "was_fallback_used" not in self.df.columns:
            return {}
        
        llm_only = self.df[self.df["was_fallback_used"] == False]
        with_fallback = self.df[self.df["was_fallback_used"] == True]
        
        return {
            "llm_only": {
                "count": len(llm_only),
                "accuracy_mean": llm_only["accuracy"].mean() if len(llm_only) > 0 else 0,
                "accuracy_std": llm_only["accuracy"].std() if len(llm_only) > 0 else 0,
                "success_rate": llm_only["is_correct"].mean() if len(llm_only) > 0 else 0,
            },
            "with_fallback": {
                "count": len(with_fallback),
                "accuracy_mean": with_fallback["accuracy"].mean() if len(with_fallback) > 0 else 0,
                "accuracy_std": with_fallback["accuracy"].std() if len(with_fallback) > 0 else 0,
                "success_rate": with_fallback["is_correct"].mean() if len(with_fallback) > 0 else 0,
            },
            "fallback_usage_rate": len(with_fallback) / max(1, len(self.df))
        }
    
    def fallback_reasons_distribution(self) -> pd.Series:
        """
        Distribution des raisons d'utilisation du fallback.
        
        Returns:
            Series avec le compte par raison
        """
        if "fallback_reason" not in self.df.columns:
            return pd.Series()
        
        return self.df["fallback_reason"].value_counts()
    
    # ==================== TIMING METRICS ====================
    
    def timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calcule les statistiques de timing.
        
        Returns:
            Dict avec les stats pour chaque composante de temps
        """
        result = {}
        
        timing_cols = [
            ("timing_total", "total"),
            ("timing_llm_response", "llm"),
            ("timing_detection", "detection"),
            ("timing_action_execution", "execution"),
            ("execution_time", "total_legacy")
        ]
        
        for col, name in timing_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    result[name] = {
                        "mean": data.mean(),
                        "std": data.std(),
                        "min": data.min(),
                        "max": data.max(),
                        "median": data.median()
                    }
        
        return result
    
    def timing_by_model(self) -> pd.DataFrame:
        """
        Temps moyen par modèle.
        
        Returns:
            DataFrame avec le temps moyen par modèle
        """
        if "model" not in self.df.columns:
            return pd.DataFrame()
        
        time_col = "timing_total" if "timing_total" in self.df.columns else "execution_time"
        if time_col not in self.df.columns:
            return pd.DataFrame()
        
        return self.df.groupby("model")[time_col].agg(["mean", "std", "count"]).round(3)
    
    # ==================== STATISTICAL TESTS ====================
    
    def compare_models_ttest(
        self, 
        model1: str, 
        model2: str, 
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Effectue un t-test entre deux modèles.
        
        Args:
            model1: Premier modèle
            model2: Deuxième modèle
            metric: Métrique à comparer
            
        Returns:
            Dict avec t-statistic, p-value, et conclusion
        """
        if "model" not in self.df.columns or metric not in self.df.columns:
            return {}
        
        data1 = self.df[self.df["model"] == model1][metric].dropna()
        data2 = self.df[self.df["model"] == model2][metric].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return {"error": "Not enough data for t-test"}
        
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        return {
            "model1": model1,
            "model2": model2,
            "metric": metric,
            "model1_mean": data1.mean(),
            "model2_mean": data2.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01
        }
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Calcule les corrélations entre variables numériques.
        
        Returns:
            Matrice de corrélation
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numeric_cols].corr().round(3)
    
    # ==================== SUMMARY STATISTICS ====================
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé complet des statistiques.
        
        Returns:
            Dict avec toutes les métriques importantes
        """
        return {
            "overall": self.overall_accuracy(),
            "by_transformation": self.accuracy_by_transformation().to_dict(),
            "by_model": self.accuracy_by_model().to_dict(),
            "llm_vs_fallback": self.llm_vs_fallback_comparison(),
            "timing": self.timing_statistics(),
            "total_tasks": len(self.df),
            "unique_models": self.df["model"].nunique() if "model" in self.df.columns else 0,
            "unique_transformations": self.df["primary_transformation"].nunique() if "primary_transformation" in self.df.columns else 0,
        }
    
    def to_latex_table(self, metric_df: pd.DataFrame, caption: str = "") -> str:
        """
        Convertit un DataFrame en tableau LaTeX.
        
        Args:
            metric_df: DataFrame à convertir
            caption: Légende du tableau
            
        Returns:
            String LaTeX
        """
        latex = metric_df.to_latex(
            float_format="%.3f",
            caption=caption,
            label=f"tab:{caption.lower().replace(' ', '_')}"
        )
        return latex

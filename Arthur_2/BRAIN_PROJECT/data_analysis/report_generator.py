"""
report_generator.py - Génération de rapports pour publications
==============================================================

Génère des rapports en différents formats (LaTeX, CSV, Markdown)
pour faciliter l'intégration dans des articles scientifiques.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class ReportGenerator:
    """
    Génère des rapports pour publications scientifiques.
    
    Example:
        from data_analysis import DataLoader, MetricsCalculator, ReportGenerator
        
        loader = DataLoader()
        df = loader.load_all_batches("results/")
        calc = MetricsCalculator(df)
        
        gen = ReportGenerator(df, calc)
        gen.generate_latex_tables("output/tables/")
        gen.generate_csv_summary("output/summary.csv")
    """
    
    def __init__(self, df: pd.DataFrame, metrics_calculator=None):
        """
        Initialise le générateur.
        
        Args:
            df: DataFrame avec les données
            metrics_calculator: Instance de MetricsCalculator (optionnel)
        """
        self.df = df
        self.metrics = metrics_calculator
    
    # ==================== LATEX EXPORT ====================
    
    def generate_latex_tables(
        self, 
        output_dir: str = "latex/",
        prefix: str = "brain"
    ):
        """
        Génère tous les tableaux LaTeX.
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Tableau d'accuracy par transformation
        if "primary_transformation" in self.df.columns:
            table = self._accuracy_by_transformation_table()
            filepath = output_path / f"{prefix}_accuracy_transformation.tex"
            with open(filepath, 'w') as f:
                f.write(table)
            print(f"Generated: {filepath}")
        
        # 2. Tableau de comparaison des modèles
        if "model" in self.df.columns and self.df["model"].nunique() > 1:
            table = self._model_comparison_table()
            filepath = output_path / f"{prefix}_model_comparison.tex"
            with open(filepath, 'w') as f:
                f.write(table)
            print(f"Generated: {filepath}")
        
        # 3. Tableau LLM vs Fallback
        if "was_fallback_used" in self.df.columns:
            table = self._llm_fallback_table()
            filepath = output_path / f"{prefix}_llm_fallback.tex"
            with open(filepath, 'w') as f:
                f.write(table)
            print(f"Generated: {filepath}")
        
        # 4. Tableau de timing
        table = self._timing_table()
        filepath = output_path / f"{prefix}_timing.tex"
        with open(filepath, 'w') as f:
            f.write(table)
        print(f"Generated: {filepath}")
    
    def _accuracy_by_transformation_table(self) -> str:
        """Génère le tableau LaTeX d'accuracy par transformation."""
        stats = self.df.groupby("primary_transformation")["accuracy"].agg(["mean", "std", "count"])
        stats["success_rate"] = self.df.groupby("primary_transformation")["is_correct"].mean()
        stats = stats.sort_values("mean", ascending=False)
        
        # Formater pour LaTeX
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Accuracy by Transformation Type}
\\label{tab:accuracy_transformation}
\\begin{tabular}{lcccc}
\\toprule
Transformation & Mean Acc. & Std & Success Rate & N \\\\
\\midrule
"""
        for idx, row in stats.iterrows():
            latex += f"{idx} & {row['mean']:.3f} & {row['std']:.3f} & {row['success_rate']:.3f} & {int(row['count'])} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex
    
    def _model_comparison_table(self) -> str:
        """Génère le tableau LaTeX de comparaison des modèles."""
        stats = self.df.groupby("model")["accuracy"].agg(["mean", "std", "count"])
        stats["success_rate"] = self.df.groupby("model")["is_correct"].mean()
        stats = stats.sort_values("mean", ascending=False)
        
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Model Performance Comparison}
\\label{tab:model_comparison}
\\begin{tabular}{lcccc}
\\toprule
Model & Mean Acc. & Std & Success Rate & N \\\\
\\midrule
"""
        for idx, row in stats.iterrows():
            latex += f"{idx} & {row['mean']:.3f} & {row['std']:.3f} & {row['success_rate']:.3f} & {int(row['count'])} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex
    
    def _llm_fallback_table(self) -> str:
        """Génère le tableau LaTeX LLM vs Fallback."""
        llm_only = self.df[self.df["was_fallback_used"] == False]
        with_fb = self.df[self.df["was_fallback_used"] == True]
        
        latex = """\\begin{table}[htbp]
\\centering
\\caption{LLM vs Fallback Performance}
\\label{tab:llm_fallback}
\\begin{tabular}{lccc}
\\toprule
Method & Mean Accuracy & Success Rate & N \\\\
\\midrule
"""
        if len(llm_only) > 0:
            latex += f"LLM Only & {llm_only['accuracy'].mean():.3f} & {llm_only['is_correct'].mean():.3f} & {len(llm_only)} \\\\\n"
        if len(with_fb) > 0:
            latex += f"With Fallback & {with_fb['accuracy'].mean():.3f} & {with_fb['is_correct'].mean():.3f} & {len(with_fb)} \\\\\n"
        
        latex += f"Overall & {self.df['accuracy'].mean():.3f} & {self.df['is_correct'].mean():.3f} & {len(self.df)} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex
    
    def _timing_table(self) -> str:
        """Génère le tableau LaTeX de timing."""
        timing_cols = {
            "timing_detection": "Detection",
            "timing_llm_response": "LLM Response",
            "timing_action_execution": "Action Execution",
            "execution_time": "Total"
        }
        
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Execution Time Analysis (seconds)}
\\label{tab:timing}
\\begin{tabular}{lccc}
\\toprule
Component & Mean & Std & Median \\\\
\\midrule
"""
        for col, name in timing_cols.items():
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    latex += f"{name} & {data.mean():.3f} & {data.std():.3f} & {data.median():.3f} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex
    
    # ==================== CSV EXPORT ====================
    
    def generate_csv_summary(
        self, 
        output_path: str = "summary.csv"
    ):
        """
        Génère un résumé CSV pour Excel/Google Sheets.
        
        Args:
            output_path: Chemin du fichier de sortie
        """
        # Créer un résumé avec les métriques principales
        summary_data = []
        
        # Global stats
        summary_data.append({
            "Category": "Overall",
            "Metric": "Total Tasks",
            "Value": len(self.df)
        })
        summary_data.append({
            "Category": "Overall",
            "Metric": "Mean Accuracy",
            "Value": f"{self.df['accuracy'].mean():.4f}"
        })
        summary_data.append({
            "Category": "Overall",
            "Metric": "Success Rate (100%)",
            "Value": f"{self.df['is_correct'].mean():.4f}"
        })
        
        # By transformation
        if "primary_transformation" in self.df.columns:
            for trans in self.df["primary_transformation"].unique():
                if pd.notna(trans):
                    trans_df = self.df[self.df["primary_transformation"] == trans]
                    summary_data.append({
                        "Category": f"Transformation: {trans}",
                        "Metric": "Mean Accuracy",
                        "Value": f"{trans_df['accuracy'].mean():.4f}"
                    })
                    summary_data.append({
                        "Category": f"Transformation: {trans}",
                        "Metric": "Count",
                        "Value": len(trans_df)
                    })
        
        # By model
        if "model" in self.df.columns:
            for model in self.df["model"].unique():
                model_df = self.df[self.df["model"] == model]
                summary_data.append({
                    "Category": f"Model: {model}",
                    "Metric": "Mean Accuracy",
                    "Value": f"{model_df['accuracy'].mean():.4f}"
                })
                summary_data.append({
                    "Category": f"Model: {model}",
                    "Metric": "Count",
                    "Value": len(model_df)
                })
        
        # Save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        print(f"Generated: {output_path}")
    
    def export_full_data(self, output_path: str = "full_data.csv"):
        """
        Exporte toutes les données vers CSV.
        
        Args:
            output_path: Chemin du fichier de sortie
        """
        self.df.to_csv(output_path, index=False)
        print(f"Exported: {output_path}")
    
    # ==================== MARKDOWN EXPORT ====================
    
    def generate_markdown_report(
        self, 
        output_path: str = "report.md"
    ):
        """
        Génère un rapport Markdown complet.
        
        Args:
            output_path: Chemin du fichier de sortie
        """
        lines = [
            "# BRAIN Project - Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"- **Total tasks analyzed:** {len(self.df)}",
            f"- **Overall accuracy:** {self.df['accuracy'].mean():.1%}",
            f"- **Success rate (100%):** {self.df['is_correct'].mean():.1%}",
            "",
        ]
        
        # Models
        if "model" in self.df.columns:
            lines.append("## Performance by Model")
            lines.append("")
            lines.append("| Model | Mean Accuracy | Std | N |")
            lines.append("|-------|---------------|-----|---|")
            
            stats = self.df.groupby("model")["accuracy"].agg(["mean", "std", "count"])
            for model, row in stats.iterrows():
                lines.append(f"| {model} | {row['mean']:.1%} | {row['std']:.3f} | {int(row['count'])} |")
            lines.append("")
        
        # Transformations
        if "primary_transformation" in self.df.columns:
            lines.append("## Performance by Transformation")
            lines.append("")
            lines.append("| Transformation | Mean Accuracy | Success Rate | N |")
            lines.append("|----------------|---------------|--------------|---|")
            
            stats = self.df.groupby("primary_transformation")["accuracy"].agg(["mean", "count"])
            stats["success_rate"] = self.df.groupby("primary_transformation")["is_correct"].mean()
            stats = stats.sort_values("mean", ascending=False)
            
            for trans, row in stats.iterrows():
                lines.append(f"| {trans} | {row['mean']:.1%} | {row['success_rate']:.1%} | {int(row['count'])} |")
            lines.append("")
        
        # LLM vs Fallback
        if "was_fallback_used" in self.df.columns:
            lines.append("## LLM vs Fallback")
            lines.append("")
            
            fallback_rate = self.df["was_fallback_used"].mean()
            llm_only = self.df[self.df["was_fallback_used"] == False]
            with_fb = self.df[self.df["was_fallback_used"] == True]
            
            lines.append(f"- **Fallback usage rate:** {fallback_rate:.1%}")
            if len(llm_only) > 0:
                lines.append(f"- **LLM-only accuracy:** {llm_only['accuracy'].mean():.1%}")
            if len(with_fb) > 0:
                lines.append(f"- **With fallback accuracy:** {with_fb['accuracy'].mean():.1%}")
            lines.append("")
        
        # Write
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Generated: {output_path}")
    
    # ==================== JSON EXPORT ====================
    
    def generate_json_summary(self, output_path: str = "summary.json"):
        """
        Génère un résumé JSON structuré.
        
        Args:
            output_path: Chemin du fichier de sortie
        """
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(self.df),
            "overall": {
                "accuracy_mean": self.df["accuracy"].mean(),
                "accuracy_std": self.df["accuracy"].std(),
                "success_rate": self.df["is_correct"].mean()
            }
        }
        
        # By transformation
        if "primary_transformation" in self.df.columns:
            by_trans = {}
            for trans in self.df["primary_transformation"].dropna().unique():
                trans_df = self.df[self.df["primary_transformation"] == trans]
                by_trans[trans] = {
                    "count": len(trans_df),
                    "accuracy_mean": trans_df["accuracy"].mean(),
                    "success_rate": trans_df["is_correct"].mean()
                }
            summary["by_transformation"] = by_trans
        
        # By model
        if "model" in self.df.columns:
            by_model = {}
            for model in self.df["model"].unique():
                model_df = self.df[self.df["model"] == model]
                by_model[model] = {
                    "count": len(model_df),
                    "accuracy_mean": model_df["accuracy"].mean(),
                    "success_rate": model_df["is_correct"].mean()
                }
            summary["by_model"] = by_model
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Generated: {output_path}")

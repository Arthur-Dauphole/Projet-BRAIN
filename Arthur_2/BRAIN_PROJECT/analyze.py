#!/usr/bin/env python3
"""
analyze.py - Script d'analyse des résultats de batch
====================================================

Usage:
    python analyze.py                           # Analyse tous les batchs
    python analyze.py --dir results/            # Spécifier le répertoire
    python analyze.py --output figures/         # Spécifier la sortie
    python analyze.py --format latex            # Générer des tableaux LaTeX
    python analyze.py --interactive             # Mode interactif (afficher les graphiques)
    
Example workflow:
    1. Lancer plusieurs batchs avec différents modèles
    2. Exécuter: python analyze.py --output analysis/
    3. Utiliser les figures et tableaux dans l'article
"""

import argparse
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from data_analysis import DataLoader, MetricsCalculator, AnalysisVisualizer, ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des résultats de batch BRAIN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                         # Analyse basique
  python analyze.py --output analysis/      # Sauvegarder dans un dossier
  python analyze.py --format all            # Générer tous les formats
  python analyze.py --interactive           # Afficher les graphiques
        """
    )
    
    parser.add_argument(
        "--dir", "-d",
        default="results/",
        help="Répertoire contenant les batchs (default: results/)"
    )
    parser.add_argument(
        "--output", "-o",
        default="analysis/",
        help="Répertoire de sortie (default: analysis/)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["figures", "latex", "csv", "markdown", "json", "all"],
        default="all",
        help="Format de sortie (default: all)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Afficher les graphiques interactivement"
    )
    parser.add_argument(
        "--pattern",
        default="batch_*",
        help="Pattern pour les dossiers de batch (default: batch_*)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BRAIN Data Analysis")
    print("=" * 60)
    print(f"\n📁 Source: {args.dir}")
    print(f"📊 Output: {args.output}")
    print(f"📄 Format: {args.format}")
    print()
    
    # === 1. CHARGER LES DONNÉES ===
    print("Loading data...")
    loader = DataLoader()
    
    try:
        df = loader.load_all_batches(args.dir, pattern=args.pattern)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if df.empty:
        print("No data found. Make sure you have run some batches first.")
        sys.exit(1)
    
    print(f"\n✓ Loaded {len(df)} task results")
    
    # Afficher un aperçu
    print("\nData overview:")
    print(f"  - Models: {df['model'].unique().tolist() if 'model' in df.columns else 'N/A'}")
    print(f"  - Transformations: {df['primary_transformation'].nunique() if 'primary_transformation' in df.columns else 'N/A'} types")
    print(f"  - Overall accuracy: {df['accuracy'].mean():.1%}")
    
    # === 2. CALCULER LES MÉTRIQUES ===
    print("\n" + "-" * 40)
    print("Calculating metrics...")
    
    calc = MetricsCalculator(df)
    
    # Afficher les métriques clés
    print("\n📊 Key Metrics:")
    
    overall = calc.overall_accuracy()
    print(f"  Overall Accuracy: {overall['mean']:.1%} (±{overall['std']:.1%})")
    
    by_trans = calc.accuracy_by_transformation()
    if not by_trans.empty:
        print("\n  By Transformation:")
        for trans, row in by_trans.head(5).iterrows():
            print(f"    {trans}: {row['mean']:.1%} (n={int(row['count'])})")
    
    llm_fb = calc.llm_vs_fallback_comparison()
    if llm_fb:
        print(f"\n  LLM vs Fallback:")
        print(f"    Fallback usage: {llm_fb['fallback_usage_rate']:.1%}")
        print(f"    LLM-only accuracy: {llm_fb['llm_only']['accuracy_mean']:.1%}")
        print(f"    With fallback: {llm_fb['with_fallback']['accuracy_mean']:.1%}")
    
    # === 3. CRÉER LA SORTIE ===
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 40)
    print("Generating outputs...")
    
    # Figures
    if args.format in ["figures", "all"]:
        print("\n📈 Generating figures...")
        figures_dir = output_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        viz = AnalysisVisualizer(df)
        
        viz.plot_accuracy_by_transformation(
            save_path=str(figures_dir / "accuracy_by_transformation.png")
        )
        
        if df["model"].nunique() > 1:
            viz.plot_model_comparison(
                save_path=str(figures_dir / "model_comparison.png")
            )
        
        viz.plot_accuracy_boxplot(
            save_path=str(figures_dir / "accuracy_boxplot.png")
        )
        
        viz.plot_timing_breakdown(
            save_path=str(figures_dir / "timing_breakdown.png")
        )
        
        viz.plot_llm_vs_fallback(
            save_path=str(figures_dir / "llm_vs_fallback.png")
        )
        
        if args.interactive:
            viz.show_all()
    
    # LaTeX
    if args.format in ["latex", "all"]:
        print("\n📝 Generating LaTeX tables...")
        latex_dir = output_path / "latex"
        
        gen = ReportGenerator(df, calc)
        gen.generate_latex_tables(str(latex_dir))
    
    # CSV
    if args.format in ["csv", "all"]:
        print("\n📄 Generating CSV files...")
        
        gen = ReportGenerator(df, calc)
        gen.generate_csv_summary(str(output_path / "summary.csv"))
        gen.export_full_data(str(output_path / "full_data.csv"))
    
    # Markdown
    if args.format in ["markdown", "all"]:
        print("\n📋 Generating Markdown report...")
        
        gen = ReportGenerator(df, calc)
        gen.generate_markdown_report(str(output_path / "report.md"))
    
    # JSON
    if args.format in ["json", "all"]:
        print("\n📊 Generating JSON summary...")
        
        gen = ReportGenerator(df, calc)
        gen.generate_json_summary(str(output_path / "summary.json"))
    
    # === 4. RÉSUMÉ ===
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved to: {output_path.absolute()}")
    
    # Lister les fichiers créés
    print("\nGenerated files:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            rel_path = f.relative_to(output_path)
            print(f"  - {rel_path}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

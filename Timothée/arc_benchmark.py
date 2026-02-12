"""
SYSTÈME DE BENCHMARK POUR LE SYSTÈME DE RÉSOLUTION DE GRILLES ARC
- Évaluation sur un grand nombre de grilles
- Statistiques de performance détaillées
- Visualisations des résultats
- Analyse par type de règles
"""

"""
SYSTÈME DE BENCHMARK POUR LE SYSTÈME DE RÉSOLUTION DE GRILLES ARC
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import time
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import sys
import copy

# Essayer d'importer les classes ARC
try:
    # Si elles sont déjà importées dans le namespace global
    AdvancedGridRuleExtractor = AdvancedGridRuleExtractor
    AdvancedGridTransformer = AdvancedGridTransformer
    GeometryDetector = GeometryDetector
    print("✓ Classes ARC disponibles")
except NameError:
    # Sinon, essayer d'importer depuis vos fichiers
    try:
        from Merge_ResolutionARC import (
            AdvancedGridRuleExtractor,
            AdvancedGridTransformer,
            GeometryDetector
        )
        print("✓ Classes ARC importées depuis Merge_ResolutionARC")
    except ImportError:
        try:
            # Essayez d'autres noms possibles
            from main_solver import (
                AdvancedGridRuleExtractor,
                AdvancedGridTransformer,
                GeometryDetector
            )
            print("✓ Classes ARC importées depuis main_solver")
        except ImportError:
            print("⚠️  Classes ARC non trouvées. Le benchmark utilisera des classes factices.")
            
            # Classes factices pour éviter les erreurs
            class GeometryDetector:
                def detect_all(self, grid):
                    return {"summary": {}, "colors_used": []}
            
            class AdvancedGridRuleExtractor:
                def __init__(self, use_shape_detection=True):
                    pass
                def extract_from_examples(self, examples):
                    pass
                def get_rules(self):
                    return {}
            
            class AdvancedGridTransformer:
                def __init__(self, rules):
                    pass
                def apply_rules(self, grid):
                    return grid
# Import des classes existantes (ajustez le chemin d'import si nécessaire)
# from votre_code_principal import AdvancedGridRuleExtractor, AdvancedGridTransformer, GeometryDetector

class ARCBatchBenchmark:
    """
    Classe principale pour le benchmarking du système ARC sur un ensemble de fichiers.
    """
    
    def __init__(self, use_shape_detection: bool = True, verbose: bool = False, mode: str = 'simple'):
        """
        Initialise le benchmark.
        
        Args:
            use_shape_detection: Si True, utilise la détection de formes
            verbose: Si True, affiche des informations détaillées pendant le traitement
        """
        self.use_shape_detection = use_shape_detection
        self.verbose = verbose
        self.mode = mode
        self.results = []
        self.statistics = {}
        
    def load_all_json_files(self, directory_path: str) -> List[Dict]:
        """
        Charge tous les fichiers JSON d'un répertoire.
        
        Args:
            directory_path: Chemin vers le dossier contenant les fichiers JSON
            
        Returns:
            Liste des données chargées avec leurs noms de fichiers
        """
        print(f"Chargement des fichiers JSON depuis {directory_path}")
        
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        if not json_files:
            print(f"ATTENTION: Aucun fichier JSON trouvé dans {directory_path}")
            return []
        
        loaded_data = []
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    filename = os.path.basename(filepath)
                    loaded_data.append({
                        'filename': filename,
                        'data': data,
                        'filepath': filepath
                    })
                if self.verbose:
                    print(f"  ✓ {filename}")
            except Exception as e:
                print(f"  ✗ Erreur lors du chargement de {filepath}: {e}")
        
        print(f"Chargement terminé: {len(loaded_data)} fichiers chargés")
        return loaded_data
    
    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """
        Évalue un seul problème ARC.
        
        Args:
            problem_data: Données du problème (avec 'train' et 'test')
            
        Returns:
            Résultats de l'évaluation pour ce problème
        """
        filename = problem_data['filename']
        data = problem_data['data']
        
        print(f"\n{'='*70}")
        print(f"ÉVALUATION DU PROBLÈME: {filename}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Vérifier la structure des données
        if 'train' not in data:
            print(f"  ✗ Fichier {filename} n'a pas de données d'entraînement ('train')")
            return {
                'filename': filename,
                'success': False,
                'error': 'Missing train data',
                'execution_time': 0
            }
        
        # Extraire les règles
        try:
            extractor = AdvancedGridRuleExtractor(use_shape_detection=self.use_shape_detection, mode=self.mode)
            extractor.extract_from_examples(data["train"])
            rules = extractor.get_rules()
            
            # Analyser les types de règles
            rule_types = self._analyze_rule_types(rules)
            
            if self.verbose:
                print(f"  Règles extraites: {rule_types}")
            
            # Préparer le transformateur
            transformer = AdvancedGridTransformer(rules)
            
            # Évaluer sur les tests
            test_results = []
            total_tests = 0
            correct_tests = 0
            
            # Chercher les tests (supporte "Test" ou "test")
            test_key = "Test" if "Test" in data else "test"
            
            if test_key in data and data[test_key]:
                for test_idx, test_case in enumerate(data[test_key]):
                    total_tests += 1
                    
                    input_grid = test_case["input"]
                    
                    # Générer la prédiction
                    predicted_output = transformer.apply_rules(input_grid)
                    
                    # Vérifier si l'output attendu existe pour évaluation
                    if "output" in test_case:
                        expected_output = test_case["output"]
                        is_correct = self._compare_grids(predicted_output, expected_output)
                        
                        if is_correct:
                            correct_tests += 1
                        
                        test_results.append({
                            'test_index': test_idx,
                            'input': input_grid,
                            'predicted': predicted_output,
                            'expected': expected_output,
                            'correct': is_correct
                        })
                        
                        if self.verbose:
                            status = "✓" if is_correct else "✗"
                            print(f"  Test {test_idx+1}: {status}")
                    else:
                        # Pas d'output attendu, on ne peut pas évaluer
                        test_results.append({
                            'test_index': test_idx,
                            'input': input_grid,
                            'predicted': predicted_output,
                            'expected': None,
                            'correct': None
                        })
            
            execution_time = time.time() - start_time
            
            # Calculer les statistiques
            accuracy = correct_tests / total_tests if total_tests > 0 else 0
            
            result = {
                'filename': filename,
                'success': True,
                'execution_time': execution_time,
                'rule_types': rule_types,
                'total_tests': total_tests,
                'correct_tests': correct_tests,
                'accuracy': accuracy,
                'test_results': test_results,
                'rules_summary': {
                    'color_changes_count': len(rules.get('color_changes', {})),
                    'connections_count': len(rules.get('connections', {})),
                    'translations_count': len(rules.get('translations', {})),
                    'propagations_count': len(rules.get('propagations', {})),
                    'position_changes_count': len(rules.get('position_based_changes', {}))
                }
            }
            
            print(f"  Résultat: {correct_tests}/{total_tests} tests corrects ({accuracy*100:.1f}%)")
            print(f"  Temps d'exécution: {execution_time:.2f} secondes")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ✗ Erreur lors du traitement: {e}")
            return {
                'filename': filename,
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _analyze_rule_types(self, rules: Dict) -> List[str]:
        """
        Analyse les types de règles extraites.
        
        Args:
            rules: Dictionnaire de règles
            
        Returns:
            Liste des types de règles détectés
        """
        rule_types = []
        
        if rules.get('color_changes'):
            rule_types.append('color_changes')
        
        if rules.get('connections'):
            rule_types.append('connections')
        
        if rules.get('translations'):
            rule_types.append('translations')
        
        if rules.get('propagations'):
            rule_types.append('propagations')
        
        if rules.get('position_based_changes'):
            rule_types.append('position_changes')
        
        # Détection de patterns spécifiques
        color_changes = rules.get('color_changes', {})
        if len(color_changes) >= 4:
            # Vérifier si c'est une bijection
            unique_values = set(color_changes.values())
            if len(color_changes) == len(unique_values):
                rule_types.append('bijection')
        
        return rule_types
    
    def _compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """
        Compare deux grilles pixel par pixel.
        
        Args:
            grid1: Première grille
            grid2: Deuxième grille
            
        Returns:
            True si les grilles sont identiques
        """
        if len(grid1) != len(grid2):
            return False
        
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        
        return True
    
    def run_benchmark(self, directory_path: str) -> None:
        """
        Exécute le benchmark sur tous les fichiers d'un répertoire.
        
        Args:
            directory_path: Chemin vers le dossier contenant les fichiers JSON
        """
        print("\n" + "="*70)
        print("DÉMARRAGE DU BENCHMARK COMPLET")
        print("="*70)
        
        # Charger tous les fichiers
        all_data = self.load_all_json_files(directory_path)
        
        if not all_data:
            print("Aucune donnée à traiter. Fin du benchmark.")
            return
        
        # Traiter chaque fichier
        total_files = len(all_data)
        successful_files = 0
        failed_files = 0
        
        self.results = []
        
        for idx, problem_data in enumerate(all_data):
            print(f"\n[{idx+1}/{total_files}] Traitement de {problem_data['filename']}")
            
            result = self.evaluate_single_problem(problem_data)
            self.results.append(result)
            
            if result['success']:
                successful_files += 1
            else:
                failed_files += 1
        
        # Générer les statistiques
        self._generate_statistics()
        
        print("\n" + "="*70)
        print("BENCHMARK TERMINÉ")
        print("="*70)
        print(f"Fichiers traités: {total_files}")
        print(f"  ✓ Succès: {successful_files}")
        print(f"  ✗ Échecs: {failed_files}")
        
        if successful_files > 0:
            print(f"\nPrécision moyenne: {self.statistics.get('overall_accuracy', 0)*100:.1f}%")
            print(f"Temps d'exécution moyen: {self.statistics.get('avg_execution_time', 0):.2f} secondes")
    
    def _generate_statistics(self) -> None:
        """Génère les statistiques à partir des résultats."""
        successful_results = [r for r in self.results if r.get('success')]
        
        if not successful_results:
            self.statistics = {
                'total_files': len(self.results),
                'successful_files': 0,
                'failed_files': len(self.results),
                'overall_accuracy': 0,
                'avg_execution_time': 0
            }
            return
        
        # Statistiques générales
        total_tests = sum(r.get('total_tests', 0) for r in successful_results)
        total_correct = sum(r.get('correct_tests', 0) for r in successful_results)
        
        # Regrouper par types de règles
        rule_type_distribution = defaultdict(int)
        rule_type_accuracy = defaultdict(list)
        
        for result in successful_results:
            rule_types = result.get('rule_types', [])
            accuracy = result.get('accuracy', 0)
            
            for rule_type in rule_types:
                rule_type_distribution[rule_type] += 1
                rule_type_accuracy[rule_type].append(accuracy)
        
        # Calculer les moyennes par type de règle
        rule_type_avg_accuracy = {}
        for rule_type, accuracies in rule_type_accuracy.items():
            rule_type_avg_accuracy[rule_type] = np.mean(accuracies) if accuracies else 0
        
        # Distribution des combinaisons de règles
        rule_combinations = Counter()
        for result in successful_results:
            rule_types = result.get('rule_types', [])
            if rule_types:
                combination_key = '+'.join(sorted(rule_types))
                rule_combinations[combination_key] += 1
        
        self.statistics = {
            'total_files': len(self.results),
            'successful_files': len(successful_results),
            'failed_files': len(self.results) - len(successful_results),
            'total_tests': total_tests,
            'total_correct': total_correct,
            'overall_accuracy': total_correct / total_tests if total_tests > 0 else 0,
            'avg_execution_time': np.mean([r.get('execution_time', 0) for r in successful_results]),
            'rule_type_distribution': dict(rule_type_distribution),
            'rule_type_accuracy': dict(rule_type_avg_accuracy),
            'rule_combinations': dict(rule_combinations),
            'individual_results': self.results
        }
    
    def generate_report(self, output_dir: str = "benchmark_results") -> None:
        """
        Génère un rapport complet avec visualisations.
        
        Args:
            output_dir: Répertoire où sauvegarder les résultats
        """
        if not self.results:
            print("Aucun résultat à reporter. Exécutez d'abord run_benchmark().")
            return
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer un timestamp pour les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les résultats bruts
        raw_results_file = os.path.join(output_dir, f"raw_results_{timestamp}.json")
        with open(raw_results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Sauvegarder les statistiques
        stats_file = os.path.join(output_dir, f"statistics_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2, default=str)
        
        print(f"\nRapport généré dans: {output_dir}")
        print(f"  - Résultats bruts: {os.path.basename(raw_results_file)}")
        print(f"  - Statistiques: {os.path.basename(stats_file)}")
        
        # Générer les visualisations
        self._generate_visualizations(output_dir, timestamp)
        
        # Générer un rapport texte
        self._generate_text_report(output_dir, timestamp)
    
    def _generate_visualizations(self, output_dir: str, timestamp: str) -> None:
        """Génère les graphiques de visualisation."""
        try:
            # 1. Graphique global de performance
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"Benchmark ARC - Résultats ({timestamp})", fontsize=16, fontweight='bold')
            
            # A. Distribution succès/échecs
            ax1 = axes[0, 0]
            success_count = self.statistics['successful_files']
            fail_count = self.statistics['failed_files']
            labels = ['Succès', 'Échecs']
            sizes = [success_count, fail_count]
            colors = ['#2ECC40', '#FF4136']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribution Succès/Échecs')
            
            # B. Précision globale
            ax2 = axes[0, 1]
            accuracy = self.statistics['overall_accuracy'] * 100
            ax2.bar(['Précision'], [accuracy], color='#0074D9')
            ax2.set_ylim([0, 100])
            ax2.set_ylabel('Précision (%)')
            ax2.set_title(f'Précision Globale: {accuracy:.1f}%')
            
            # Ajouter la valeur sur la barre
            ax2.text(0, accuracy + 1, f'{accuracy:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
            
            # C. Distribution des types de règles
            ax3 = axes[0, 2]
            rule_dist = self.statistics.get('rule_type_distribution', {})
            if rule_dist:
                labels = list(rule_dist.keys())
                values = list(rule_dist.values())
                
                # Traductions françaises
                french_labels = {
                    'color_changes': 'Changements\ncouleur',
                    'connections': 'Connexions',
                    'translations': 'Translations',
                    'propagations': 'Propagations',
                    'position_changes': 'Par position',
                    'bijection': 'Bijection'
                }
                
                display_labels = [french_labels.get(label, label) for label in labels]
                
                bars = ax3.bar(display_labels, values, color='#FF851B')
                ax3.set_title('Distribution des Types de Règles')
                ax3.set_ylabel('Nombre de problèmes')
                
                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
            
            # D. Précision par type de règle
            ax4 = axes[1, 0]
            rule_accuracy = self.statistics.get('rule_type_accuracy', {})
            if rule_accuracy:
                labels = list(rule_accuracy.keys())
                accuracies = [acc * 100 for acc in rule_accuracy.values()]
                
                display_labels = [french_labels.get(label, label) for label in labels]
                
                bars = ax4.bar(display_labels, accuracies, color='#F012BE')
                ax4.set_title('Précision par Type de Règles')
                ax4.set_ylabel('Précision (%)')
                ax4.set_ylim([0, 100])
                
                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom')
            
            # E. Temps d'exécution
            ax5 = axes[1, 1]
            execution_times = [r.get('execution_time', 0) for r in self.results if r.get('success')]
            if execution_times:
                ax5.hist(execution_times, bins=20, color='#7FDBFF', edgecolor='black')
                ax5.set_title('Distribution des Temps d\'Exécution')
                ax5.set_xlabel('Temps (secondes)')
                ax5.set_ylabel('Nombre de problèmes')
                
                # Ajouter des lignes verticales pour la moyenne
                avg_time = np.mean(execution_times)
                ax5.axvline(avg_time, color='red', linestyle='--', linewidth=2,
                          label=f'Moyenne: {avg_time:.2f}s')
                ax5.legend()
            
            # F. Combinaisons de règles les plus fréquentes
            ax6 = axes[1, 2]
            rule_combo = self.statistics.get('rule_combinations', {})
            if rule_combo:
                # Prendre les 10 combinaisons les plus fréquentes
                top_combos = sorted(rule_combo.items(), key=lambda x: x[1], reverse=True)[:10]
                combo_labels = [combo[0] for combo in top_combos]
                combo_counts = [combo[1] for combo in top_combos]
                
                # Raccourcir les labels pour une meilleure lisibilité
                short_labels = []
                for label in combo_labels:
                    short_label = label
                    if len(label) > 20:
                        short_label = label[:17] + '...'
                    short_labels.append(short_label)
                
                y_pos = np.arange(len(short_labels))
                ax6.barh(y_pos, combo_counts, color='#2ECC40')
                ax6.set_yticks(y_pos)
                ax6.set_yticklabels(short_labels)
                ax6.invert_yaxis()  # Plus fréquent en haut
                ax6.set_title('Top 10 Combinaisons de Règles')
                ax6.set_xlabel('Fréquence')
            
            plt.tight_layout()
            
            # Sauvegarder le graphique principal
            main_plot_file = os.path.join(output_dir, f"main_plot_{timestamp}.png")
            plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
            print(f"  - Graphique principal: {os.path.basename(main_plot_file)}")
            
            # 2. Graphique détaillé des performances individuelles
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            
            successful_results = [r for r in self.results if r.get('success')]
            if successful_results:
                filenames = [r['filename'] for r in successful_results]
                accuracies = [r.get('accuracy', 0) * 100 for r in successful_results]
                
                # Trier par précision
                sorted_data = sorted(zip(filenames, accuracies), key=lambda x: x[1], reverse=True)
                sorted_filenames, sorted_accuracies = zip(*sorted_data) if sorted_data else ([], [])
                
                # Limiter à 50 fichiers pour la lisibilité
                if len(sorted_filenames) > 50:
                    sorted_filenames = sorted_filenames[:50]
                    sorted_accuracies = sorted_accuracies[:50]
                    note = "(Top 50 seulement)"
                else:
                    note = ""
                
                bars = ax2.bar(range(len(sorted_filenames)), sorted_accuracies, color='#0074D9')
                ax2.set_xlabel('Problèmes')
                ax2.set_ylabel('Précision (%)')
                ax2.set_title(f'Performance par Problème {note}')
                ax2.set_xticks(range(len(sorted_filenames)))
                
                # Raccourcir les noms de fichiers pour l'axe x
                short_names = []
                for name in sorted_filenames:
                    if len(name) > 15:
                        short_names.append(name[:12] + '...')
                    else:
                        short_names.append(name)
                
                ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
                
                # Colorer les barres selon la précision
                for i, bar in enumerate(bars):
                    if sorted_accuracies[i] >= 80:
                        bar.set_color('#2ECC40')  # Vert pour haute précision
                    elif sorted_accuracies[i] >= 50:
                        bar.set_color('#FFDC00')  # Jaune pour précision moyenne
                    else:
                        bar.set_color('#FF4136')  # Rouge pour basse précision
                
                # Ligne de moyenne
                avg_accuracy = np.mean(sorted_accuracies) if sorted_accuracies else 0
                ax2.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2,
                          label=f'Moyenne: {avg_accuracy:.1f}%')
                ax2.legend()
            
            plt.tight_layout()
            detail_plot_file = os.path.join(output_dir, f"detail_plot_{timestamp}.png")
            plt.savefig(detail_plot_file, dpi=300, bbox_inches='tight')
            print(f"  - Graphique détaillé: {os.path.basename(detail_plot_file)}")
            
            plt.close('all')
            
        except Exception as e:
            print(f"  ✗ Erreur lors de la génération des visualisations: {e}")
    
    def _generate_text_report(self, output_dir: str, timestamp: str) -> None:
        """Génère un rapport texte détaillé."""
        report_file = os.path.join(output_dir, f"report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAPPORT DE BENCHMARK ARC\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Paramètres: use_shape_detection={self.use_shape_detection}\n\n")
            
            # Résumé global
            f.write("RÉSUMÉ GLOBAL\n")
            f.write("-"*70 + "\n")
            f.write(f"Fichiers traités: {self.statistics['total_files']}\n")
            f.write(f"  ✓ Succès: {self.statistics['successful_files']}\n")
            f.write(f"  ✗ Échecs: {self.statistics['failed_files']}\n")
            f.write(f"Tests totaux: {self.statistics['total_tests']}\n")
            f.write(f"Tests corrects: {self.statistics['total_correct']}\n")
            f.write(f"Précision globale: {self.statistics['overall_accuracy']*100:.2f}%\n")
            f.write(f"Temps d'exécution moyen: {self.statistics['avg_execution_time']:.2f} secondes\n\n")
            
            # Distribution des types de règles
            f.write("DISTRIBUTION DES TYPES DE RÈGLES\n")
            f.write("-"*70 + "\n")
            rule_dist = self.statistics.get('rule_type_distribution', {})
            if rule_dist:
                for rule_type, count in sorted(rule_dist.items(), key=lambda x: x[1], reverse=True):
                    accuracy = self.statistics['rule_type_accuracy'].get(rule_type, 0) * 100
                    f.write(f"{rule_type:20s}: {count:3d} problèmes ({accuracy:6.2f}% de précision)\n")
            f.write("\n")
            
            # Meilleures et pires performances
            successful_results = [r for r in self.results if r.get('success')]
            if successful_results:
                f.write("TOP 10 MEILLEURES PERFORMANCES\n")
                f.write("-"*70 + "\n")
                sorted_by_accuracy = sorted(successful_results, 
                                          key=lambda x: x.get('accuracy', 0), 
                                          reverse=True)[:10]
                
                for i, result in enumerate(sorted_by_accuracy, 1):
                    filename = result['filename']
                    accuracy = result.get('accuracy', 0) * 100
                    tests = result.get('total_tests', 0)
                    correct = result.get('correct_tests', 0)
                    time_taken = result.get('execution_time', 0)
                    
                    f.write(f"{i:2d}. {filename:30s} {correct:2d}/{tests:2d} ({accuracy:6.1f}%) "
                          f"en {time_taken:.2f}s\n")
                
                f.write("\nTOP 10 PIRE PERFORMANCES\n")
                f.write("-"*70 + "\n")
                sorted_by_accuracy_worst = sorted(successful_results, 
                                                key=lambda x: x.get('accuracy', 0))[:10]
                
                for i, result in enumerate(sorted_by_accuracy_worst, 1):
                    filename = result['filename']
                    accuracy = result.get('accuracy', 0) * 100
                    tests = result.get('total_tests', 0)
                    correct = result.get('correct_tests', 0)
                    
                    f.write(f"{i:2d}. {filename:30s} {correct:2d}/{tests:2d} ({accuracy:6.1f}%)\n")
            
            # Analyse des échecs
            failed_results = [r for r in self.results if not r.get('success')]
            if failed_results:
                f.write("\nANALYSE DES ÉCHECS\n")
                f.write("-"*70 + "\n")
                for result in failed_results:
                    filename = result['filename']
                    error = result.get('error', 'Unknown error')
                    f.write(f"• {filename}: {error}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("FIN DU RAPPORT\n")
            f.write("="*70 + "\n")
        
        print(f"  - Rapport texte: {os.path.basename(report_file)}")
    
    def get_summary(self) -> Dict:
        """Retourne un résumé des statistiques."""
        return {
            'overall_accuracy': self.statistics.get('overall_accuracy', 0),
            'success_rate': self.statistics.get('successful_files', 0) / 
                          max(1, self.statistics.get('total_files', 1)),
            'avg_execution_time': self.statistics.get('avg_execution_time', 0),
            'total_tests': self.statistics.get('total_tests', 0),
            'total_correct': self.statistics.get('total_correct', 0)
        }
    def generate_visualizations(self, output_dir: str = "visualizations") -> None:
        """
        Génère des visualisations complètes de tous les tests.
        
        Args:
            output_dir: Dossier pour sauvegarder les visualisations
        """
        from arc_benchmark import generate_comprehensive_visualization
        
        generate_comprehensive_visualization(self, output_dir)
    
    def generate_grid_visualizations(self, output_dir="grid_visualizations"):
        """Génère des visualisations des grilles."""
        print(f"\nGénération des visualisations de grilles dans {output_dir}...")
        
        try:
            # Essayer d'appeler la fonction autonome définie ci-dessous
            generate_grid_visualizations_function(self, output_dir)
        except NameError:
            # Si la fonction n'existe pas, créer un fallback simple
            import os
            import matplotlib.pyplot as plt
            import numpy as np
            
            os.makedirs(output_dir, exist_ok=True)
            
            successful_results = [r for r in self.results if r.get('success')]
            print(f"Problèmes à visualiser: {len(successful_results)}")
            
            if not successful_results:
                print("Aucun résultat à visualiser.")
                return
            
            # Fonction de conversion simple
            def grid_to_rgb(grid):
                arc_colors = {
                    0: (0, 0, 0),        # noir
                    1: (255, 255, 255),  # blanc
                    2: (255, 0, 0),      # rouge
                    3: (0, 255, 0),      # vert
                    4: (0, 0, 255),      # bleu
                    5: (255, 255, 0),    # jaune
                    6: (255, 0, 255),    # magenta
                    7: (0, 255, 255),    # cyan
                    8: (128, 128, 128),  # gris
                    9: (255, 128, 0)     # orange
                }
                
                h = len(grid)
                w = len(grid[0])
                rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
                
                for i in range(h):
                    for j in range(w):
                        color_val = grid[i][j]
                        rgb_array[i, j] = arc_colors.get(color_val, (0, 0, 0))
                
                return rgb_array
            
            # Visualisation simple pour 5 premiers problèmes
            for idx, result in enumerate(successful_results[:5]):
                filename = result['filename']
                accuracy = result.get('accuracy', 0) * 100
                
                # Prendre le premier test
                test_results = result.get('test_results', [])
                if not test_results:
                    continue
                    
                test_result = test_results[0]
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle(f'{filename} - Précision: {accuracy:.1f}%')
                
                # Input
                if 'input' in test_result:
                    axes[0].imshow(grid_to_rgb(test_result['input']))
                    axes[0].set_title('Input')
                    axes[0].axis('off')
                
                # Prédit
                if 'predicted' in test_result:
                    axes[1].imshow(grid_to_rgb(test_result['predicted']))
                    axes[1].set_title('Prédit')
                    axes[1].axis('off')
                
                # Attendu
                if 'expected' in test_result and test_result['expected'] is not None:
                    axes[2].imshow(grid_to_rgb(test_result['expected']))
                    axes[2].set_title('Attendu')
                    axes[2].axis('off')
                else:
                    axes[2].text(0.5, 0.5, 'Pas de référence', 
                            ha='center', va='center')
                    axes[2].axis('off')
                
                plt.tight_layout()
                safe_name = filename.replace('.json', '').replace(' ', '_')
                save_path = os.path.join(output_dir, f"{safe_name}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ {filename} → {save_path}")
    """
GÉNÉRATEUR D'IMAGE GÉANTE - Toutes les grilles de tous les problèmes dans une seule image
"""
    def load_arc_colors():
        """Retourne la palette de couleurs ARC standard."""
        return {
            0: [0, 0, 0],         # noir
            1: [255, 255, 255],   # blanc
            2: [255, 0, 0],       # rouge
            3: [0, 255, 0],       # vert
            4: [0, 0, 255],       # bleu
            5: [255, 255, 0],     # jaune
            6: [255, 0, 255],     # magenta
            7: [0, 255, 255],     # cyan
            8: [128, 128, 128],   # gris
            9: [255, 128, 0]      # orange
        }

    def grid_to_image(grid, arc_colors):
        """Convertit une grille ARC en image numpy RGB."""
        h = len(grid)
        w = len(grid[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                color_val = grid[i][j]
                if color_val in arc_colors:
                    img[i, j] = arc_colors[color_val]
                else:
                    img[i, j] = [0, 0, 0]  # noir par défaut
        
        return img

    def create_giant_visualization(benchmark_results, output_dir="giant_visualization", max_problems=50):
        """
        Crée une seule image géante qui montre TOUTES les grilles de TOUS les problèmes.
        Chaque problème a sa propre section avec :
        - Tous les exemples d'entraînement (input/output)
        - Tous les tests (input/prédiction/expected)
        
        Args:
            benchmark_results: Liste des résultats du benchmark
            output_dir: Dossier de sortie
            max_problems: Nombre maximum de problèmes à inclure
        """
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        arc_colors = load_arc_colors()
        
        # Filtrer les résultats réussis
        successful_results = [r for r in benchmark_results if r.get('success')]
        
        if not successful_results:
            print("Aucun résultat réussi à visualiser.")
            return
        
        print(f"Préparation de l'image géante avec {min(len(successful_results), max_problems)} problèmes...")
        
        # Limiter le nombre de problèmes
        results_to_show = successful_results[:max_problems]
        
        # Calculer le nombre total de grilles pour déterminer la taille de l'image
        total_grids = 0
        problem_info = []
        
        for result in results_to_show:
            filename = result['filename']
            accuracy = result.get('accuracy', 0) * 100
            
            # Charger les données originales
            try:
                if 'filepath' in result:
                    filepath = result['filepath']
                else:
                    filepath = f"Arthur_2/BRAIN_PROJECT/data/{filename}"
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        original_data = json.load(f)
                    
                    train_examples = original_data.get('train', [])
                    test_examples = original_data.get('test', [])
                    
                    # Nombre de grilles pour ce problème
                    n_train_grids = len(train_examples) * 2  # input + output
                    n_test_grids = len(test_examples) * 3    # input + prédiction + expected
                    
                    total_grids += n_train_grids + n_test_grids
                    
                    problem_info.append({
                        'filename': filename,
                        'accuracy': accuracy,
                        'train_examples': train_examples,
                        'test_examples': test_examples,
                        'test_results': result.get('test_results', []),
                        'total_grids': n_train_grids + n_test_grids
                    })
                else:
                    print(f"  ⚠️ Fichier non trouvé: {filepath}")
            except Exception as e:
                print(f"  ⚠️ Erreur avec {filename}: {e}")
        
        if total_grids == 0:
            print("Aucune donnée à afficher.")
            return
        
        print(f"Total de grilles à afficher: {total_grids}")
        
        # Calculer la disposition optimale
        # On veut environ 8 colonnes pour une bonne lisibilité
        n_cols = 8
        n_rows = (total_grids + n_cols - 1) // n_cols
        
        # Créer une figure GÉANTE
        # Taille basée sur le nombre de lignes et colonnes
        fig_width = n_cols * 2.5  # 2.5 pouces par colonne
        fig_height = n_rows * 2.5  # 2.5 pouces par ligne
        
        # Limiter la taille maximum
        fig_width = min(fig_width, 50)
        fig_height = min(fig_height, 100)
        
        print(f"Création de l'image géante: {fig_width:.1f} x {fig_height:.1f} pouces")
        print(f"Disposition: {n_rows} lignes x {n_cols} colonnes")
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Titre principal
        avg_accuracy = np.mean([p['accuracy'] for p in problem_info])
        plt.suptitle(f"VISUALISATION COMPLÈTE DE TOUS LES PROBLÈMES ARC\n"
                    f"{len(problem_info)} problèmes - Précision moyenne: {avg_accuracy:.1f}%\n"
                    f"Chaque problème montre: [Entraînement: Input→Output] [Tests: Input→Prédiction→Expected]",
                    fontsize=24, fontweight='bold', y=0.99)
        
        # Utiliser GridSpec pour un contrôle précis
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Position courante dans la grille
        current_row = 0
        current_col = 0
        
        # Pour chaque problème
        for problem_idx, problem in enumerate(problem_info):
            filename = problem['filename']
            accuracy = problem['accuracy']
            
            # Titre du problème (en texte dans la première cellule)
            if current_col < n_cols:
                ax = fig.add_subplot(gs[current_row, current_col])
                ax.text(0.5, 0.5, f"{filename}\n{accuracy:.1f}%", 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
                ax.axis('off')
                current_col += 1
            
            # Afficher les exemples d'entraînement
            for i, example in enumerate(problem['train_examples']):
                if current_col >= n_cols:
                    current_col = 0
                    current_row += 1
                
                # Input
                if current_row < n_rows:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    if 'input' in example:
                        img = grid_to_image(example['input'], arc_colors)
                        ax.imshow(img)
                        ax.set_title(f"Train {i+1} In", fontsize=6)
                    ax.axis('off')
                    current_col += 1
                
                if current_col >= n_cols:
                    current_col = 0
                    current_row += 1
                
                # Output
                if current_row < n_rows:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    if 'output' in example:
                        img = grid_to_image(example['output'], arc_colors)
                        ax.imshow(img)
                        ax.set_title(f"Train {i+1} Out", fontsize=6, color='blue')
                    ax.axis('off')
                    current_col += 1
            
            # Afficher les tests avec prédictions
            test_results = problem['test_results']
            for i in range(len(problem['test_examples'])):
                # Input du test
                if current_col >= n_cols:
                    current_col = 0
                    current_row += 1
                
                if current_row < n_rows:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    if i < len(problem['test_examples']) and 'input' in problem['test_examples'][i]:
                        img = grid_to_image(problem['test_examples'][i]['input'], arc_colors)
                        ax.imshow(img)
                        ax.set_title(f"Test {i+1} In", fontsize=6)
                    ax.axis('off')
                    current_col += 1
                
                # Prédiction
                if current_col >= n_cols:
                    current_col = 0
                    current_row += 1
                
                if current_row < n_rows:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    if i < len(test_results) and 'predicted' in test_results[i]:
                        img = grid_to_image(test_results[i]['predicted'], arc_colors)
                        ax.imshow(img)
                        
                        # Vérifier si correct
                        is_correct = test_results[i].get('correct', False)
                        status = "✓" if is_correct else "✗"
                        color = 'green' if is_correct else 'red'
                        ax.set_title(f"Test {i+1} Préd {status}", fontsize=6, color=color)
                    ax.axis('off')
                    current_col += 1
                
                # Expected
                if current_col >= n_cols:
                    current_col = 0
                    current_row += 1
                
                if current_row < n_rows:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    if i < len(problem['test_examples']) and 'output' in problem['test_examples'][i]:
                        img = grid_to_image(problem['test_examples'][i]['output'], arc_colors)
                        ax.imshow(img)
                        ax.set_title(f"Test {i+1} Exp", fontsize=6, color='purple')
                    ax.axis('off')
                    current_col += 1
            
            # Ajouter une ligne vide entre les problèmes pour la lisibilité
            if current_col < n_cols and current_col > 0:
                # Remplir le reste de la ligne avec des cellules vides
                while current_col < n_cols:
                    ax = fig.add_subplot(gs[current_row, current_col])
                    ax.axis('off')
                    current_col += 1
            
            current_col = 0
            current_row += 1
            
            # Afficher la progression
            if (problem_idx + 1) % 10 == 0:
                print(f"  Progression: {problem_idx + 1}/{len(problem_info)} problèmes traités")
        
        # Ajuster la disposition
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        
        # Sauvegarder l'image géante
        output_path = os.path.join(output_dir, "GIANT_VISUALIZATION_ALL_PROBLEMS.png")
        print(f"\nSauvegarde de l'image géante... (cela peut prendre un moment)")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Image géante sauvegardée: {output_path}")
        print(f"Taille: {fig_width:.1f} x {fig_height:.1f} pouces")
        
        # Créer aussi une version PDF (meilleure pour le zoom)
        output_path_pdf = os.path.join(output_dir, "GIANT_VISUALIZATION_ALL_PROBLEMS.pdf")
        print("Création de la version PDF...")
        
        # Réduire la taille pour le PDF
        fig_width_pdf = min(fig_width, 30)
        fig_height_pdf = min(fig_height, 60)
        
        fig2 = plt.figure(figsize=(fig_width_pdf, fig_height_pdf))
        plt.suptitle(f"VISUALISATION COMPLÈTE - {len(problem_info)} problèmes ARC", 
                    fontsize=16, fontweight='bold', y=0.99)
        
        # Recréer avec moins de détails pour le PDF
        current_row = 0
        current_col = 0
        n_cols_pdf = 6
        
        for problem in problem_info:
            # Uniquement les prédictions principales
            if current_row < 100:  # Limiter à 100 lignes
                if problem['test_results']:
                    test = problem['test_results'][0]
                    if 'predicted' in test:
                        ax = fig2.add_subplot(gs[current_row, current_col])
                        img = grid_to_image(test['predicted'], arc_colors)
                        ax.imshow(img)
                        
                        # Titre court
                        short_name = problem['filename'][:15] + '...' if len(problem['filename']) > 15 else problem['filename']
                        ax.set_title(f"{short_name}\n{problem['accuracy']:.1f}%", fontsize=6)
                        ax.axis('off')
                        
                        current_col += 1
                        if current_col >= n_cols_pdf:
                            current_col = 0
                            current_row += 1
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"✅ Version PDF sauvegardée: {output_path_pdf}")
        print(f"\n📊 Résumé:")
        print(f"   - Problèmes inclus: {len(problem_info)}")
        print(f"   - Précision moyenne: {avg_accuracy:.1f}%")
        print(f"   - Fichiers PNG et PDF créés dans: {os.path.abspath(output_dir)}")
        print(f"   - Ouvrez le fichier PNG avec un visualiseur d'images qui supporte le zoom")

# ============================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# ============================================

def generate_comprehensive_visualization(benchmark_instance, output_dir="visualizations"):
    """
    Génère une visualisation complète de TOUS les tests dans une image géante.
    
    Args:
        benchmark_instance: Instance de ARCBatchBenchmark après exécution
        output_dir: Dossier de sortie pour les visualisations
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from datetime import datetime
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrer les résultats réussis
    successful_results = [r for r in benchmark_instance.results if r.get('success')]
    
    if not successful_results:
        print("Aucun résultat réussi à visualiser.")
        return
    
    print(f"Génération de la visualisation complète pour {len(successful_results)} problèmes...")
    
    # Pour chaque problème réussi, créer une visualisation
    for result_idx, result in enumerate(successful_results):
        try:
            filename = result['filename']
            test_results = result.get('test_results', [])
            
            if not test_results:
                continue
            
            # Compter combien de tests ont un output attendu
            tests_with_expected = [t for t in test_results if t.get('expected') is not None]
            
            if not tests_with_expected:
                continue
            
            # Déterminer la disposition de la grille
            n_tests = len(tests_with_expected)
            n_cols = min(3, n_tests)  # Max 3 colonnes
            n_rows = (n_tests + n_cols - 1) // n_cols
            
            # Créer une figure par problème
            fig = plt.figure(figsize=(n_cols * 10, n_rows * 6))
            fig.suptitle(f'Problème: {filename}\nPrécision: {result.get("accuracy", 0)*100:.1f}%', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Ajouter chaque test
            for test_idx, test_result in enumerate(tests_with_expected):
                ax = plt.subplot(n_rows, n_cols, test_idx + 1)
                
                input_grid = test_result['input']
                predicted_grid = test_result['predicted']
                expected_grid = test_result['expected']
                is_correct = test_result['correct']
                
                # Afficher les trois grilles côte à côte
                display_grid = []
                
                # Combiner les grilles horizontalement avec une colonne de séparation
                h = len(input_grid)
                w = len(input_grid[0])
                
                # Créer une grille combinée: INPUT | PREDICTED | EXPECTED
                combined_grid = []
                for i in range(h):
                    row = []
                    # Input
                    row.extend(input_grid[i])
                    # Séparateur (couleur spéciale, par exemple 10)
                    row.append(10)
                    # Predicted
                    row.extend(predicted_grid[i])
                    # Séparateur
                    row.append(10)
                    # Expected
                    row.extend(expected_grid[i])
                    combined_grid.append(row)
                
                # Convertir en tableau RGB
                from arc_system_complet import grid_to_rgb, ColorMapper
                
                # Étendre la ColorMapper pour le séparateur
                class ExtendedColorMapper(ColorMapper):
                    COLOR_MAP = ColorMapper.COLOR_MAP.copy()
                    COLOR_MAP[10] = "#FFFFFF"  # Blanc pour les séparateurs
                
                # Fonction de conversion étendue
                def extended_grid_to_rgb(grid):
                    h, w = len(grid), len(grid[0])
                    rgb_array = np.zeros((h, w, 3))
                    
                    for y in range(h):
                        for x in range(w):
                            hex_color = ExtendedColorMapper.hex(grid[y][x])
                            hex_color = hex_color.lstrip('#')
                            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 
                                       for i in (0, 2, 4))
                            rgb_array[y, x] = rgb
                    
                    return rgb_array
                
                # Afficher la grille combinée
                ax.imshow(extended_grid_to_rgb(combined_grid))
                
                # Ajouter les titres et séparateurs
                ax.set_title(f"Test {test_idx + 1} - {'✓' if is_correct else '✗'}", 
                           fontsize=12, fontweight='bold')
                
                # Ajouter les labels
                ax.text(w/2, -0.5, "INPUT", ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
                ax.text(w + 0.5 + w/2, -0.5, "PRÉDIT", ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', 
                       color='green' if is_correct else 'red')
                ax.text(2*w + 1.5 + w/2, -0.5, "ATTENDU", ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
                
                # Ajouter les séparateurs verticaux
                ax.axvline(x=w + 0.5, color='white', linewidth=3)
                ax.axvline(x=2*w + 1.5, color='white', linewidth=3)
                
                # Supprimer les axes
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                # Cadre de couleur selon la correction
                for spine in ax.spines.values():
                    spine.set_edgecolor('green' if is_correct else 'red')
                    spine.set_linewidth(3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Sauvegarder
            safe_filename = filename.replace('.json', '').replace(' ', '_')
            output_path = os.path.join(output_dir, f"visualisation_{safe_filename}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ {filename} visualisé → {output_path}")
            
        except Exception as e:
            print(f"  ✗ Erreur lors de la visualisation de {result['filename']}: {e}")
            continue
    
    # Créer une visualisation RÉSUMÉ de tous les problèmes
    print("\nGénération de la visualisation récapitulative...")
    generate_summary_visualization(benchmark_instance, output_dir)
    
    print(f"\nVisualisations sauvegardées dans: {os.path.abspath(output_dir)}")


def generate_grid_visualizations(benchmark_instance, output_dir="grid_visualizations"):
    """
    Génère des images montrant les grilles d'entrée, prédites et attendues.
    
    Args:
        benchmark_instance: Instance de ARCBatchBenchmark après exécution
        output_dir: Dossier de sortie
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import json
    
    # Créer le dossier
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrer les résultats avec tests réussis pour démonstration
    successful_results = [r for r in benchmark_instance.results if r.get('success')]
    
    print(f"\n{'='*60}")
    print("GÉNÉRATION DES VISUALISATIONS DE GRILLES")
    print(f"{'='*60}")
    print(f"Problèmes à visualiser: {len(successful_results)}")
    
    # Limiter à 20 problèmes pour éviter trop d'images
    if len(successful_results) > 20:
        print("Limitation à 20 problèmes (les plus intéressants)...")
        # Trier par précision (mélanger succès/échecs)
        successful_results.sort(key=lambda x: abs(x.get('accuracy', 0) - 0.5))
        successful_results = successful_results[:20]
    
    # Créer une fonction de conversion de grille vers RGB
    def grid_to_rgb(grid):
        """Convertit une grille ARC (valeurs 0-9) en image RGB."""
        # Palette de couleurs ARC standard
        arc_colors = {
            0: (0, 0, 0),        # noir
            1: (255, 255, 255),  # blanc
            2: (255, 0, 0),      # rouge
            3: (0, 255, 0),      # vert
            4: (0, 0, 255),      # bleu
            5: (255, 255, 0),    # jaune
            6: (255, 0, 255),    # magenta
            7: (0, 255, 255),    # cyan
            8: (128, 128, 128),  # gris
            9: (255, 128, 0)     # orange
        }
        
        h = len(grid)
        w = len(grid[0])
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                color_val = grid[i][j]
                rgb_array[i, j] = arc_colors.get(color_val, (0, 0, 0))
        
        return rgb_array
    
    for result_idx, result in enumerate(successful_results):
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        test_results = result.get('test_results', [])
        
        if not test_results:
            continue
        
        # Créer une figure pour ce problème
        n_tests = len(test_results)
        fig, axes = plt.subplots(n_tests, 3, figsize=(12, 4 * n_tests))
        if n_tests == 1:
            axes = [axes]  # Pour gérer le cas d'un seul test
        
        fig.suptitle(f'{filename}\nPrécision: {accuracy:.1f}%', 
                    fontsize=16, fontweight='bold')
        
        for test_idx, test_result in enumerate(test_results):
            input_grid = test_result['input']
            predicted_grid = test_result['predicted']
            expected_grid = test_result.get('expected')
            is_correct = test_result.get('correct', False)
            
            # TITRES
            if expected_grid is not None:
                titles = ['Input', 'Prédit', 'Attendu']
                colors = ['black', 'green' if is_correct else 'red', 'blue']
                grids = [input_grid, predicted_grid, expected_grid]
            else:
                titles = ['Input', 'Prédit', 'Pas de référence']
                colors = ['black', 'orange', 'gray']
                grids = [input_grid, predicted_grid, None]
            
            # AFFICHER CHAQUE GRILLE
            for col_idx in range(3):
                ax = axes[test_idx][col_idx] if n_tests > 1 else axes[col_idx]
                
                if grids[col_idx] is not None:
                    # Convertir et afficher la grille
                    rgb_image = grid_to_rgb(grids[col_idx])
                    ax.imshow(rgb_image)
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Configuration des axes
                ax.set_title(titles[col_idx], color=colors[col_idx], 
                           fontweight='bold', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Cadre coloré
                for spine in ax.spines.values():
                    spine.set_edgecolor(colors[col_idx])
                    spine.set_linewidth(3)
            
            # Ajouter un indicateur de résultat
            if expected_grid is not None:
                result_text = '✓ CORRECT' if is_correct else '✗ INCORRECT'
                result_color = 'green' if is_correct else 'red'
                fig.text(0.5, 0.95 - (test_idx * 0.8/n_tests), 
                        f"Test {test_idx+1}: {result_text}",
                        ha='center', color=result_color, fontweight='bold',
                        fontsize=14, transform=fig.transFigure)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarder
        safe_name = filename.replace('.json', '').replace(' ', '_')
        save_path = os.path.join(output_dir, f"{safe_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {filename} → {save_path}")
    
    # Créer une image RÉCAPITULATIVE avec miniatures
    print("\nGénération de l'image récapitulative...")
    
    if successful_results:
        # Trier par précision
        successful_results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Créer une grille de miniatures
        n_results = min(25, len(successful_results))  # Max 25
        n_cols = 5
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, result in enumerate(successful_results[:n_results]):
            ax = axes[idx]
            
            # Prendre le premier test comme exemple
            test_results = result.get('test_results', [])
            if not test_results:
                continue
            
            test_result = test_results[0]
            predicted_grid = test_result['predicted']
            
            # Afficher la prédiction
            rgb_image = grid_to_rgb(predicted_grid)
            ax.imshow(rgb_image)
            
            # Titre avec nom de fichier et précision
            filename = result['filename']
            accuracy = result.get('accuracy', 0) * 100
            short_name = filename[:15] + '...' if len(filename) > 15 else filename
            
            # Couleur selon la précision
            color = 'green' if accuracy >= 80 else 'orange' if accuracy >= 50 else 'red'
            
            ax.set_title(f"{short_name}\n{accuracy:.1f}%", 
                        fontsize=9, color=color, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Cadre coloré
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
        
        # Cacher les axes non utilisés
        for idx in range(n_results, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Récapitulatif des prédictions - Précision globale: {benchmark_instance.statistics.get('overall_accuracy', 0)*100:.1f}%",
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarder
        save_path = os.path.join(output_dir, "recap_all_predictions.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Récapitulatif → {save_path}")
    
    print(f"\n✅ Visualisations sauvegardées dans: {os.path.abspath(output_dir)}")

def generate_summary_grid(benchmark_instance, output_dir):
    """Génère une image récapitulative avec miniatures."""
    import matplotlib.pyplot as plt
    import os
    
    successful_results = [r for r in benchmark_instance.results if r.get('success')]
    
    if not successful_results:
        return
    
    # Trier par précision
    successful_results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
    
    # Créer une grille de miniatures
    n_results = min(25, len(successful_results))  # Max 25
    n_cols = 5
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, result in enumerate(successful_results[:n_results]):
        ax = axes[idx]
        
        # Prendre le premier test comme exemple
        test_results = result.get('test_results', [])
        if not test_results:
            continue
        
        test_result = test_results[0]
        predicted_grid = test_result['predicted']
        
        # Afficher la prédiction
        try:
            from Merge_ResolutionARC import grid_to_rgb
            ax.imshow(grid_to_rgb(predicted_grid))
        except:
            ax.imshow(predicted_grid, cmap='viridis')
        
        # Titre avec nom de fichier et précision
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        short_name = filename[:15] + '...' if len(filename) > 15 else filename
        
        # Couleur selon la précision
        color = 'green' if accuracy >= 80 else 'orange' if accuracy >= 50 else 'red'
        
        ax.set_title(f"{short_name}\n{accuracy:.1f}%", 
                    fontsize=9, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Cadre coloré
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Cacher les axes non utilisés
    for idx in range(n_results, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Récapitulatif des prédictions - Précision globale: {benchmark_instance.statistics.get('overall_accuracy', 0)*100:.1f}%",
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarder
    save_path = os.path.join(output_dir, "recap_all_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Récapitulatif → {save_path}")

def generate_summary_visualization(benchmark_instance, output_dir):
    """
    Génère une visualisation récapitulative de TOUS les problèmes.
    """
    import matplotlib.pyplot as plt
    import os
    
    successful_results = [r for r in benchmark_instance.results if r.get('success')]
    
    if not successful_results:
        return
    
    # Trier par précision
    successful_results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
    
    # Créer une grande figure
    n_problems = len(successful_results)
    n_cols = 4  # 4 problèmes par ligne
    n_rows = (n_problems + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, result in enumerate(successful_results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        filename = result['filename']
        accuracy = result.get('accuracy', 0) * 100
        
        # Prendre le premier test comme exemple
        test_results = result.get('test_results', [])
        if not test_results:
            continue
        
        first_test = test_results[0]
        if first_test.get('expected') is None:
            continue
        
        input_grid = first_test['input']
        h, w = len(input_grid), len(input_grid[0])
        
        # Afficher seulement l'input pour le résumé
        from arc_system_complet import grid_to_rgb
        ax.imshow(grid_to_rgb(input_grid))
        
        # Titre avec précision
        color = 'green' if accuracy >= 80 else 'orange' if accuracy >= 50 else 'red'
        ax.set_title(f"{filename[:15]}...\n{accuracy:.1f}%", 
                    fontsize=9, color=color, fontweight='bold')
        
        # Supprimer les axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Cadre selon la précision
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Cacher les axes non utilisés
    for idx in range(len(successful_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Récapitulatif des {n_problems} problèmes - Précision globale: "
                f"{benchmark_instance.statistics.get('overall_accuracy', 0)*100:.1f}%", 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarder
    output_path = os.path.join(output_dir, "recapitulatif_complet.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Récapitulatif généré → {output_path}")

def run_comprehensive_benchmark(data_directory: str, 
                               output_directory: str = "benchmark_results",
                               use_shape_detection: bool = True,
                               verbose: bool = False) -> Dict:
    """
    Fonction principale pour exécuter un benchmark complet.
    
    Args:
        data_directory: Répertoire contenant les fichiers JSON de test
        output_directory: Répertoire pour sauvegarder les résultats
        use_shape_detection: Si True, active la détection de formes
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        Statistiques du benchmark
    """
    print("\n" + "="*70)
    print("BENCHMARK COMPLET DU SYSTÈME ARC")
    print("="*70)
    
    # Créer l'instance du benchmark
    benchmark = ARCBatchBenchmark(
        use_shape_detection=use_shape_detection,
        verbose=verbose
    )
    
    # Exécuter le benchmark
    benchmark.run_benchmark(data_directory)
    
    # Générer le rapport
    benchmark.generate_report(output_directory)
    
    # Retourner les statistiques
    return benchmark.get_summary()

def compare_configurations(data_directory: str, 
                         configs: List[Dict],
                         output_directory: str = "comparison_results") -> None:
    """
    Compare différentes configurations du système.
    
    Args:
        data_directory: Répertoire contenant les fichiers JSON
        configs: Liste de configurations à tester
                Exemple: [{'name': 'Avec détection formes', 'use_shape_detection': True},
                         {'name': 'Sans détection formes', 'use_shape_detection': False}]
        output_directory: Répertoire pour sauvegarder les résultats
    """
    print("\n" + "="*70)
    print("COMPARAISON DE CONFIGURATIONS")
    print("="*70)
    
    results = []
    
    for config in configs:
        config_name = config.get('name', 'Configuration')
        use_shape_detection = config.get('use_shape_detection', True)
        
        print(f"\nConfiguration: {config_name}")
        print(f"  use_shape_detection: {use_shape_detection}")
        
        # Exécuter le benchmark pour cette configuration
        benchmark = ARCBatchBenchmark(
            use_shape_detection=use_shape_detection,
            verbose=False
        )
        
        benchmark.run_benchmark(data_directory)
        
        # Collecter les résultats
        summary = benchmark.get_summary()
        summary['config_name'] = config_name
        summary['use_shape_detection'] = use_shape_detection
        
        results.append(summary)
    
    # Générer un graphique de comparaison
    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparaison des Configurations ({timestamp})", fontsize=16, fontweight='bold')
    
    # Extraire les données
    config_names = [r['config_name'] for r in results]
    accuracies = [r['overall_accuracy'] * 100 for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]
    exec_times = [r['avg_execution_time'] for r in results]
    
    # Graphique 1: Précision
    ax1 = axes[0, 0]
    bars1 = ax1.bar(config_names, accuracies, color='#0074D9')
    ax1.set_ylabel('Précision (%)')
    ax1.set_title('Précision Globale')
    ax1.set_ylim([0, 100])
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Graphique 2: Taux de succès
    ax2 = axes[0, 1]
    bars2 = ax2.bar(config_names, success_rates, color='#2ECC40')
    ax2.set_ylabel('Taux de succès (%)')
    ax2.set_title('Taux de Succès (fichiers)')
    ax2.set_ylim([0, 100])
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Graphique 3: Temps d'exécution
    ax3 = axes[1, 0]
    bars3 = ax3.bar(config_names, exec_times, color='#FF851B')
    ax3.set_ylabel('Temps (secondes)')
    ax3.set_title('Temps d\'Exécution Moyen')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Graphique 4: Score composite
    ax4 = axes[1, 1]
    # Score = 0.5 * précision + 0.3 * succès + 0.2 * (1 - temps_normalisé)
    max_time = max(exec_times) if exec_times else 1
    scores = []
    
    for i, r in enumerate(results):
        time_score = 1 - (exec_times[i] / max_time)
        composite_score = 0.5 * r['overall_accuracy'] + 0.3 * r['success_rate'] + 0.2 * time_score
        scores.append(composite_score * 100)
    
    bars4 = ax4.bar(config_names, scores, color='#F012BE')
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Score Composite')
    ax4.set_ylim([0, 100])
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Sauvegarder
    comparison_file = os.path.join(output_directory, f"comparison_{timestamp}.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    
    # Sauvegarder les données
    data_file = os.path.join(output_directory, f"comparison_data_{timestamp}.json")
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparaison sauvegardée dans: {output_directory}")
    print(f"  - Graphique: {os.path.basename(comparison_file)}")
    print(f"  - Données: {os.path.basename(data_file)}")
    
    # Afficher le gagnant
    if scores:
        best_idx = np.argmax(scores)
        print(f"\nMEILLEURE CONFIGURATION: {config_names[best_idx]}")
        print(f"  Score: {scores[best_idx]:.1f}%")
        print(f"  Précision: {accuracies[best_idx]:.1f}%")
        print(f"  Taux de succès: {success_rates[best_idx]:.1f}%")
        print(f"  Temps d'exécution: {exec_times[best_idx]:.2f}s")

# ============================================
# POINT D'ENTRÉE PRINCIPAL POUR LE BENCHMARK
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark du système ARC")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Répertoire contenant les fichiers JSON de test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Répertoire pour sauvegarder les résultats")
    parser.add_argument("--no-shape-detection", action="store_true",
                       help="Désactiver la détection de formes")
    parser.add_argument("--verbose", action="store_true",
                       help="Afficher des informations détaillées")
    parser.add_argument("--compare", action="store_true",
                       help="Exécuter une comparaison de configurations")
    
    args = parser.parse_args()
    
    if args.compare:
        # Mode comparaison
        configs = [
            {'name': 'Avec détection formes', 'use_shape_detection': True},
            {'name': 'Sans détection formes', 'use_shape_detection': False}
        ]
        
        compare_configurations(
            data_directory=args.data_dir,
            configs=configs,
            output_directory=args.output_dir
        )
    else:
        # Mode benchmark simple
        results = run_comprehensive_benchmark(
            data_directory=args.data_dir,
            output_directory=args.output_dir,
            use_shape_detection=not args.no_shape_detection,
            verbose=args.verbose
        )
        
        print(f"\nRésumé final:")
        print(f"  Précision: {results['overall_accuracy']*100:.1f}%")
        print(f"  Taux de succès: {results['success_rate']*100:.1f}%")
        print(f"  Temps moyen: {results['avg_execution_time']:.2f}s")





# Initialiser le benchmark
benchmark = ARCBatchBenchmark(use_shape_detection=True, verbose=False, mode = "simple")

# Exécuter sur un dossier de fichiers JSON
benchmark.run_benchmark("C:\\Users\\timor\\.virtual_documents\\Projet-BRAIN\\Arthur_2\\BRAIN_PROJECT\\data")

# Générer un rapport
benchmark.generate_report("resultats_benchmark/")

# Obtenir un résumé
summary = benchmark.get_summary()
print(f"Précision globale: {summary['overall_accuracy']*100:.1f}%")

# 🧠 BRAIN Project

**B**ridging **R**easoning and **A**I with **I**ntelligent **N**euro-symbolic Systems

Un solveur neuro-symbolique pour les puzzles [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus).

> **Version:** 2.3.0  
> **Dernière mise à jour:** Janvier 2026

---

## 📋 Table des matières

- [Description](#description)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Évaluation batch](#évaluation-batch)
- [Comparaison de modèles](#comparaison-de-modèles)
- [Structure du projet](#structure-du-projet)
- [Format des données](#format-des-données)
- [Exemples](#exemples)

---

## Description

BRAIN combine :
- **Perception symbolique** : Détection automatique de formes géométriques (carrés, rectangles, lignes, formes en L/T/+, blobs)
- **Détection de transformations** : Identification automatique des règles (translation, rotation, réflexion, changement de couleur, tiling, etc.)
- **Raisonnement LLM** : Utilisation d'un modèle de langage local (Ollama) pour inférer les règles
- **Exécution symbolique** : Application des transformations sur les grilles
- **Évaluation batch** : Exécution et analyse de multiples tâches
- **Comparaison de modèles** : Benchmark de différents LLMs sur les mêmes tâches

### Pipeline

```
Input Grid → Perception → Transformation Detection → Prompting → LLM → Execution → Analysis → Visualization
```

---

## Prérequis

### 1. Python 3.10+

Vérifiez votre version :
```bash
python3 --version
```

### 2. Ollama (LLM local)

Installez Ollama depuis [ollama.ai](https://ollama.ai/) :

**macOS :**
```bash
brew install ollama
```

**Linux :**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows :**
Téléchargez depuis [ollama.ai/download](https://ollama.ai/download)

### 3. Télécharger un modèle LLM

Lancez Ollama et téléchargez le modèle `llama3` :
```bash
# Démarrer le service Ollama (si pas déjà lancé)
ollama serve &

# Télécharger le modèle (environ 4.7 GB)
ollama pull llama3
```

---

## Installation

### 1. Cloner/accéder au projet

```bash
cd chemin/vers/BRAIN_PROJECT
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
```

### 3. Activer l'environnement

**macOS/Linux :**
```bash
source venv/bin/activate
```

**Windows :**
```bash
venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Commande de base

```bash
python main.py --task data/mock_task.json
```

### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--task FICHIER` | Chemin vers le fichier JSON de la tâche | - |
| `--batch DIR` | Lancer un batch sur toutes les tâches d'un répertoire | - |
| `--model MODELE` | Nom du modèle Ollama | `llama3` |
| `--limit N` | Limiter le nombre de tâches (batch) | Toutes |
| `--no-viz` | Désactiver la visualisation graphique | `False` |
| `--quiet` | Mode silencieux (moins de logs) | `False` |
| `--self-correct` | Activer la boucle d'auto-correction | `False` |
| `--demo` | Exécuter une démo avec données d'exemple | - |

### Exemples de commandes

```bash
# Résoudre une tâche avec visualisation
python main.py --task data/mock_task.json

# Sans visualisation (plus rapide)
python main.py --task data/mock_task.json --no-viz

# Avec un autre modèle
python main.py --task data/mock_task.json --model mistral

# Mode silencieux
python main.py --task data/mock_task.json --quiet --no-viz

# Avec auto-correction (retry si erreur)
python main.py --task data/mock_task.json --self-correct
```

---

## Évaluation batch

Exécutez plusieurs tâches et collectez des statistiques :

```bash
# Toutes les tâches du dossier data/
python main.py --batch data/

# Limité à 10 tâches
python main.py --batch data/ --limit 10

# Avec un modèle spécifique
python main.py --batch data/ --model mistral

# Résultats dans un dossier personnalisé
python main.py --batch data/ --output results_mistral/
```

**Résultats générés :**
- `summary.json` - Statistiques agrégées
- `tasks.csv` - Résultats par tâche
- `images/` - Visualisations de chaque tâche

---

## Comparaison de modèles

Comparez les performances de plusieurs LLMs :

```bash
# Lister les modèles recommandés
python compare_models.py --list-models

# Comparer llama3 et mistral sur 10 tâches
python compare_models.py --models llama3 mistral --limit 10

# Avec génération de graphiques
python compare_models.py --models llama3 mistral --visualize

# Générer les graphiques depuis des résultats existants
python compare_models.py --viz-only comparison_results/
```

**Important :** `compare_models.py` utilise exactement le même pipeline que `main.py --batch`, garantissant des résultats 100% cohérents.

### Modèles recommandés

| Modèle | Description | Taille | Installation |
|--------|-------------|--------|--------------|
| `llama3` | Meta Llama 3 8B - Bon généraliste | 4.7 GB | `ollama pull llama3` |
| `mistral` | Mistral 7B - Excellent raisonnement | 4.1 GB | `ollama pull mistral` |
| `phi3` | Microsoft Phi-3 - Petit mais capable | 2.2 GB | `ollama pull phi3` |

### Visualisations générées

- `accuracy_comparison.png` - Barplot accuracy par modèle
- `time_comparison.png` - Temps de réponse moyen
- `accuracy_vs_time.png` - Trade-off accuracy/temps
- `summary_dashboard.png` - Dashboard complet

---

## Structure du projet

```
BRAIN_PROJECT/
│
├── 📂 data/                              # 53 puzzles ARC au format JSON
│   ├── task_translation_*.json           # 8 tâches de translation
│   ├── task_rotation_*.json              # 7 tâches de rotation
│   ├── task_reflection_*.json            # 6 tâches de réflexion
│   ├── task_color_change_*.json          # 6 tâches de changement de couleur
│   ├── task_draw_line_*.json             # 5 tâches de tracé de ligne
│   ├── task_add_border_*.json            # 4 tâches d'ajout de contour
│   ├── task_tiling_*.json                # 3 tâches de pavage
│   ├── task_composite_*.json             # 3 tâches de transformations composées
│   ├── task_blob_*.json                  # 4 tâches sur formes irrégulières
│   └── task_multi_objects*.json          # 2 tâches multi-objets
│
├── 📂 modules/                           # Pipeline principal (12 modules)
│   ├── __init__.py                       # Exports publics
│   ├── types.py                          # Structures de données (Grid, ARCTask)
│   ├── detector.py                       # Perception : détection de formes
│   ├── transformation_detector.py        # Analyse : détection de transformations
│   ├── prompt_maker.py                   # Génération de prompts LLM
│   ├── llm_client.py                     # Communication Ollama (parsing JSON)
│   ├── executor.py                       # Exécution des actions DSL
│   ├── analyzer.py                       # Évaluation des résultats
│   ├── visualizer.py                     # Visualisation matplotlib
│   ├── batch_runner.py                   # Évaluation batch de tâches
│   ├── model_comparator.py               # Comparaison de modèles + graphiques
│   ├── logger.py                         # Logging structuré (TIER 1)
│   └── rule_memory.py                    # Mémoire RAG de règles (TIER 3)
│
├── 📂 data_analysis/                     # Outils d'analyse scientifique
│   ├── __init__.py                       # Exports
│   ├── data_loader.py                    # Chargement résultats batch
│   ├── metrics.py                        # Calcul de métriques statistiques
│   ├── visualizer.py                     # Graphiques IEEE/LaTeX
│   └── report_generator.py               # Génération rapports (Markdown, LaTeX)
│
├── 📂 notebooks/                         # Jupyter notebooks
│   └── analysis_example.ipynb            # Exemple d'analyse de données
│
├── 📂 results/                           # [Généré] Résultats single/batch
├── 📂 comparison_results/                # [Généré] Résultats comparaison modèles
├── 📂 analysis/                          # [Généré] Figures et rapports
│
├── 🐍 main.py                            # Point d'entrée (single + batch)
├── 🐍 compare_models.py                  # CLI comparaison de modèles
├── 🐍 analyze.py                         # CLI analyse de données
│
├── 📋 requirements.txt                   # Dépendances Python
├── 📋 CAPABILITIES.md                    # Documentation technique détaillée
└── 📋 README.md                          # Ce fichier
```

### Description des modules principaux

| Module | Rôle |
|--------|------|
| `detector.py` | Identifie les formes (carrés, rectangles, L, T, blobs...) |
| `transformation_detector.py` | Détecte les règles entre input/output |
| `executor.py` | Applique les transformations (translate, rotate, etc.) |
| `batch_runner.py` | Exécute et agrège plusieurs tâches |
| `model_comparator.py` | Compare les performances de plusieurs LLMs |

---

## Format des données

Les fichiers de tâches suivent le format officiel ARC-AGI :

```json
{
  "train": [
    {
      "input": [[0, 0, 2], [0, 2, 2], [0, 0, 0]],
      "output": [[0, 0, 0], [0, 0, 2], [0, 2, 2]]
    }
  ],
  "test": [
    {
      "input": [[2, 2, 0], [2, 0, 0], [0, 0, 0]],
      "output": [[0, 0, 0], [2, 2, 0], [2, 0, 0]]
    }
  ]
}
```

### Palette de couleurs ARC

| Code | Couleur |
|------|---------|
| 0 | Noir (fond) |
| 1 | Bleu |
| 2 | Rouge |
| 3 | Vert |
| 4 | Jaune |
| 5 | Gris |
| 6 | Magenta |
| 7 | Orange |
| 8 | Cyan |
| 9 | Marron |

---

## Exemples

### Exemple 1 : Translation simple

La tâche `mock_task.json` incluse déplace un carré rouge de 3 pixels vers la droite.

```bash
python main.py --task data/mock_task.json --no-viz
```

**Sortie attendue :**
```
STEP 1b: TRANSFORMATION DETECTION
  Example 1: [100%] Translation: dx=3 (right), dy=0 (down)
  Example 2: [100%] Translation: dx=3 (right), dy=0 (down)

STEP 5: ANALYSIS (Evaluation)
  ✓ Correct: True
  📊 Accuracy: 100.00%
```

### Exemple 2 : Utiliser vos propres puzzles

1. Créez un fichier JSON dans `data/` suivant le format ARC
2. Exécutez :
```bash
python main.py --task data/votre_puzzle.json
```

---

## Dépannage

### Erreur : "Ollama not installed"

```bash
pip install ollama
```

### Erreur : "Connection refused" 

Ollama n'est pas lancé :
```bash
ollama serve
```

### Erreur : "Model not found"

Téléchargez le modèle :
```bash
ollama pull llama3
```

### Visualisation bloquée / crash matplotlib

Utilisez l'option `--no-viz` :
```bash
python main.py --task data/mock_task.json --no-viz
```

---

## Documentation

Consultez `CAPABILITIES.md` pour la liste complète des fonctionnalités :
- Formes détectées
- Transformations supportées
- Actions disponibles
- Métriques d'évaluation

---

## Auteurs

Projet BRAIN - ISAE-SUPAERO

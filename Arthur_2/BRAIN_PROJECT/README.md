# 🧠 BRAIN Project

**B**ridging **R**easoning and **A**I with **I**ntelligent **N**euro-symbolic Systems

Un solveur neuro-symbolique pour les puzzles [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus).

---

## 📋 Table des matières

- [Description](#description)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Format des données](#format-des-données)
- [Exemples](#exemples)

---

## Description

BRAIN combine :
- **Perception symbolique** : Détection automatique de formes géométriques (carrés, rectangles, lignes, formes en L/T/+, etc.)
- **Détection de transformations** : Identification automatique des règles (translation, rotation, réflexion, changement de couleur)
- **Raisonnement LLM** : Utilisation d'un modèle de langage local (Ollama) pour inférer les règles
- **Exécution symbolique** : Application des transformations sur les grilles

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
| `--model MODELE` | Nom du modèle Ollama | `llama3` |
| `--no-viz` | Désactiver la visualisation graphique | `False` |
| `--quiet` | Mode silencieux (moins de logs) | `False` |
| `--demo` | Exécuter une démo avec données d'exemple | - |

### Exemples de commandes

```bash
# Résoudre une tâche avec visualisation
python main.py --task data/mock_task.json

# Sans visualisation (plus rapide)
python main.py --task data/mock_task.json --no-viz

# Avec un autre modèle
python main.py --task data/mock_task.json --model llama3.2

# Mode silencieux
python main.py --task data/mock_task.json --quiet --no-viz
```

### Utilisation avec l'environnement virtuel (sans l'activer)

```bash
./venv/bin/python main.py --task data/mock_task.json --no-viz
```

---

## Structure du projet

```
BRAIN_PROJECT/
│
├── data/                       # Données d'entrée (puzzles ARC)
│   └── mock_task.json          # Exemple de tâche
│
├── modules/                    # Modules du pipeline
│   ├── __init__.py             # Exports
│   ├── types.py                # Classes de données (Grid, ARCTask)
│   ├── detector.py             # Détection de formes
│   ├── transformation_detector.py  # Détection de transformations
│   ├── prompt_maker.py         # Génération de prompts
│   ├── llm_client.py           # Communication avec Ollama
│   ├── executor.py             # Exécution des actions
│   ├── analyzer.py             # Analyse des résultats
│   └── visualizer.py           # Visualisation matplotlib
│
├── main.py                     # Point d'entrée principal
├── requirements.txt            # Dépendances Python
├── CAPABILITIES.md             # Documentation des capacités
└── README.md                   # Ce fichier
```

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

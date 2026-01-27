# BRAIN Project - Capacités du Système

> **Dernière mise à jour :** Janvier 2026  
> **Version :** 1.5.0

---

## 📋 Vue d'ensemble

BRAIN est un solveur neuro-symbolique pour les puzzles ARC-AGI. Il combine :
- **Perception symbolique** : Détection et classification de formes géométriques
- **Raisonnement LLM** : Inférence de règles de transformation via un modèle de langage
- **Exécution symbolique** : Application des transformations sur les grilles

### Pipeline

```
Input Grid → Perception → Prompting → LLM Reasoning → Execution → Analysis → Visualization
```

---

## 🔍 Module DETECTOR (Perception)

### Formes détectées

| Forme | Description | Status |
|-------|-------------|--------|
| `point` | Pixel isolé (1 pixel) | ✅ |
| `horizontal_line` | Ligne horizontale (height=1, width>1) | ✅ |
| `vertical_line` | Ligne verticale (width=1, height>1) | ✅ |
| `square` | Carré plein (width=height, plein) | ✅ |
| `rectangle` | Rectangle plein (width≠height, plein) | ✅ |
| `hollow_rectangle` | Rectangle creux (cadre) | ✅ |
| `L_shape` | Forme en L | ✅ |
| `T_shape` | Forme en T | ✅ |
| `plus_shape` | Forme en + | ✅ |
| `diagonal_line` | Ligne diagonale | ✅ |
| `blob` | Forme irrégulière générique | ✅ |
| `blob_compact` | Blob rond/compact (compactness > 0.7) | ✅ |
| `blob_elongated` | Blob allongé (aspect ratio > 2.5) | ✅ |
| `blob_sparse` | Blob dispersé (density < 0.4) | ✅ |
| `blob_complex` | Blob complexe (> 6 corners) | ✅ |
| `blob_with_hole` | Blob avec trou interne | ✅ |

### Propriétés extraites

| Propriété | Description | Status |
|-----------|-------------|--------|
| `color` | Couleur (0-9) | ✅ |
| `color_name` | Nom de la couleur | ✅ |
| `bounding_box` | (min_row, min_col, max_row, max_col) | ✅ |
| `width`, `height` | Dimensions | ✅ |
| `area` | Nombre de pixels | ✅ |
| `is_filled` | Objet plein ou creux | ✅ |
| `density` | area / (width × height) | ✅ |
| `is_convex` | Forme convexe | ✅ |
| `has_hole` | Contient un trou | ✅ |
| `perimeter` | Nombre de pixels en bordure | ✅ |
| `compactness` | Circularité (4π×Area/Perimeter²) | ✅ |
| `corner_count` | Nombre de coins détectés | ✅ |
| `orientation` | horizontal/vertical/diagonal/symmetric | ✅ |
| `aspect_ratio` | width/height | ✅ |
| `shape_signature` | Signature binaire normalisée (pour comparaison) | ✅ |
| `centroid` | Centre de masse (row, col) | ✅ |

### Patterns globaux détectés

| Pattern | Description | Status |
|---------|-------------|--------|
| Symétrie horizontale | Grille symétrique haut/bas | ✅ |
| Symétrie verticale | Grille symétrique gauche/droite | ✅ |
| Symétrie diagonale | Grille symétrique selon diagonale | ✅ |
| Couleur de fond | Couleur la plus fréquente | ✅ |

---

## 🔄 Module TRANSFORMATION DETECTOR

### Transformations détectées automatiquement

| Transformation | Description | Status |
|----------------|-------------|--------|
| `translation` | Déplacement (dx, dy) | ✅ |
| `rotation` | Rotation 90°, 180°, 270° | ✅ |
| `reflection` | Miroir horizontal/vertical/diagonal | ✅ |
| `color_change` | Changement de couleur | ✅ |
| `scaling` | Agrandissement/réduction | ✅ |
| `draw_line` | Tracer une ligne entre 2 points | ✅ |
| `blob_transformation` | Transformation de formes irrégulières | ✅ |
| `translation_and_color` | Translation + changement de couleur combinés | ✅ |

### Détails de la détection de blobs

Le système peut détecter des transformations appliquées à des formes irrégulières (blobs) :

1. **Comparaison par signature** : Les blobs sont normalisés et comparés pixel par pixel
2. **Détection de rotation** : Vérifie si le blob a été pivoté de 90°, 180° ou 270°
3. **Détection de réflexion** : Vérifie si le blob a été reflété horizontalement ou verticalement
4. **Détection de translation** : Calcule le déplacement (dx, dy) entre les positions

---

## ⚡ Module EXECUTOR (Actions)

### Actions supportées

| Action | Paramètres | Description | Status |
|--------|------------|-------------|--------|
| `translate` | `dx`, `dy`, `color_filter` | Déplace les pixels | ✅ |
| `fill` | `color`, `region` | Remplit une zone | ✅ |
| `replace_color` | `from_color`, `to_color` | Change une couleur | ✅ |
| `copy` | `dx`, `dy`, `color_filter` | Copie avec offset | ✅ |
| `color_change` | `from_color`, `to_color` | Changement de couleur | ✅ |
| `rotate` | `angle`, `color_filter` | Rotation 90°/180°/270° | ✅ |
| `reflect` | `axis`, `color_filter` | Réflexion (miroir) | ✅ |
| `scale` | `factor`, `color_filter` | Agrandir/réduire | ✅ |
| `draw_line` | `color_filter` ou `point1`, `point2` | Tracer une ligne entre 2 points | ✅ |

### Détails des axes de réflexion

| Axe | Description |
|-----|-------------|
| `horizontal` | Miroir haut-bas (flipud) |
| `vertical` | Miroir gauche-droite (fliplr) |
| `diagonal_main` | Miroir diagonale principale |
| `diagonal_anti` | Miroir anti-diagonale |

### Détails de l'action draw_line

L'action `draw_line` trace une ligne entre deux points de même couleur en utilisant l'algorithme de Bresenham.

**Modes d'utilisation :**
1. **Auto-détection** : Si `color_filter` est spécifié, trouve automatiquement les 2 pixels de cette couleur et les relie
2. **Points explicites** : Utilise `point1` et `point2` dans les paramètres

**Exemple JSON :**
```json
{
  "action": "draw_line",
  "color_filter": 2
}
```
ou
```json
{
  "action": "draw_line",
  "params": {
    "point1": {"row": 2, "col": 1},
    "point2": {"row": 2, "col": 7},
    "color": 2
  }
}
```

**Types de lignes supportées :**
- Horizontale (même ligne)
- Verticale (même colonne)
- Diagonale (algorithme de Bresenham)

---

## 📊 Module ANALYZER (Évaluation)

### Métriques calculées

| Métrique | Description | Status |
|----------|-------------|--------|
| `is_correct` | Correspondance exacte | ✅ |
| `pixel_accuracy` | % de pixels corrects | ✅ |
| `iou_per_color` | IoU par couleur | ✅ |
| `shape_match` | Dimensions correctes | ✅ |
| `error_pattern` | Type d'erreur | ✅ |
| `color_confusion` | Matrice de confusion | ✅ |

---

## 🎨 Module VISUALIZER

### Visualisations disponibles

| Visualisation | Description | Status |
|---------------|-------------|--------|
| `show_grid` | Affiche une grille | ✅ |
| `show_pair` | Input/Output côte à côte | ✅ |
| `show_comparison` | Predicted vs Expected avec diff | ✅ |
| `show_task` | Tâche complète avec exemples | ✅ |
| `show_analysis_dashboard` | Dashboard d'analyse | ✅ |
| `show_color_legend` | Palette ARC | ✅ |

---

## 🧠 Module LLM CLIENT

### Capacités

| Fonctionnalité | Description | Status |
|----------------|-------------|--------|
| Connexion Ollama | Communication avec LLM local | ✅ |
| Extraction JSON | Parse les actions depuis la réponse | ✅ |
| Extraction de grille | Parse les grilles depuis la réponse | ✅ |
| Extraction du raisonnement | Isole l'explication | ✅ |

### Modèles testés

| Modèle | Status |
|--------|--------|
| `llama3` | ✅ Fonctionne |
| `llama3.2` | ✅ Fonctionne |

---

## 📝 Historique des versions

### v1.5.0 (Janvier 2026) - Batch Evaluation Mode
- ✅ **NOUVEAU: Mode Batch** (`--batch DIR`) pour évaluer plusieurs tâches automatiquement
- ✅ **Module BatchRunner** - Exécute toutes les tâches et collecte des statistiques
- ✅ **Dossiers horodatés** - Chaque batch crée `results/batch_YYYYMMDD_HHMMSS/`
- ✅ **Rapports multiples** - `summary.json`, `tasks.csv`, `README.txt`
- ✅ **Exécution non-bloquante** - Visualisations désactivées automatiquement en batch
- ✅ **Statistiques agrégées** - Accuracy moyenne, temps d'exécution, comptage des transformations
- ✅ Options: `--limit`, `--pattern`, `--output`

### v1.4.0 (Janvier 2026) - Blob Support Avancé
- ✅ **NOUVEAU: Sous-types de blobs** - `blob_compact`, `blob_elongated`, `blob_sparse`, `blob_complex`, `blob_with_hole`
- ✅ **Propriétés avancées** - `perimeter`, `compactness`, `corner_count`, `orientation`, `aspect_ratio`, `shape_signature`
- ✅ **Détection de transformation de blobs** - Translation, rotation, réflexion, changement de couleur
- ✅ **Comparaison de formes** - `compare_shapes()`, `find_matching_object()` dans SymbolDetector
- ✅ Fichiers de test: `task_blob_translation.json`, `task_blob_rotation.json`, `task_blob_reflection.json`, `task_blob_color_change.json`

### v1.3.0 (Janvier 2026) - Draw Line Support
- ✅ **NOUVEAU: Action draw_line** - Tracer une ligne entre 2 points
- ✅ Détection automatique de la transformation `draw_line` dans `TransformationDetector`
- ✅ Algorithme de Bresenham pour les lignes diagonales
- ✅ Support des lignes horizontales, verticales et diagonales
- ✅ Fichier de test: `task_draw_line.json`

### v1.2.0 (Janvier 2026) - Multi-Transform Support
- ✅ **NOUVEAU: Mode Multi-Transform** (`--multi`) pour transformations différentes par couleur
- ✅ Détection de transformations par couleur (`detect_per_color_transformations`)
- ✅ Prompts spécialisés pour multi-transform
- ✅ Parser multi-actions dans LLMClient
- ✅ Executor multi-actions (`execute_multi_actions`)
- ✅ Fichiers de test: `task_multi_objects_same_transform.json`, `task_challenge_multi_transform.json`

### v1.1.0 (Janvier 2026)
- ✅ Amélioration du système de prompt avec "DETECTED TRANSFORMATION" explicite
- ✅ Correction de la détection de translation (ignore dx=0, dy=0)
- ✅ Amélioration de la détection de rotation d'objets individuels
- ✅ Actions `rotate`, `reflect`, `scale` fonctionnelles dans l'executor
- ✅ Support de la détection de rotation pour objets de couleurs différentes

### v1.0.0 (Janvier 2026)
- ✅ Pipeline complet fonctionnel
- ✅ Détection de formes basiques et avancées
- ✅ Détection automatique des transformations
- ✅ Action `translate` fonctionnelle
- ✅ Intégration Ollama/LLama3
- ✅ Visualisation matplotlib

---

## 🚀 Roadmap

### Prochaines fonctionnalités

- [ ] Détection de patterns répétitifs
- [ ] Détection de sous-grilles
- [x] ~~Mode batch pour évaluer plusieurs tâches~~ ✅ v1.5.0
- [x] ~~Export des résultats en JSON~~ ✅ v1.5.0
- [ ] Support de transformations composées (translation + rotation simultanées)
- [ ] Auto-détection du mode (single vs multi-transform)
- [ ] Parallélisation des évaluations batch

---

## ⚠️ Limitations connues

| Limitation | Description |
|------------|-------------|
| Couleurs différentes entre exemples | En mode standard, si chaque exemple a une couleur différente, utiliser `--multi` |
| Transformations composées | Une seule transformation par couleur en mode multi |
| Taille de grille variable | Non supporté actuellement |
| Dépendance LLM | Le mode multi nécessite que le LLM retourne le bon format JSON |

---

## 🔧 Modes d'utilisation

### Mode Standard (défaut)
```bash
python main.py --task data/task.json
```
Applique la MÊME transformation à TOUS les objets.

### Mode Multi-Transform
```bash
python main.py --task data/task.json --multi
```
Applique des transformations DIFFÉRENTES à chaque COULEUR.

### Mode Batch (Évaluation en lot)
```bash
# Exécuter toutes les tâches dans data/
python main.py --batch data/

# Limiter à 10 tâches
python main.py --batch data/ --limit 10

# Filtrer par pattern
python main.py --batch data/ --pattern "task_blob_*.json"

# Spécifier le dossier de sortie
python main.py --batch data/ --output results/

# Combiner avec mode multi-transform
python main.py --batch data/ --multi --limit 5
```

#### Statistiques collectées

| Métrique | Description |
|----------|-------------|
| `total_tasks` | Nombre total de tâches |
| `successful_tasks` | Tâches exécutées sans erreur |
| `correct_tasks` | Tâches avec 100% d'accuracy |
| `overall_accuracy` | Accuracy moyenne sur toutes les tâches |
| `avg_time_per_task` | Temps moyen par tâche |
| `transformation_counts` | Comptage par type de transformation |
| `action_counts` | Comptage par action exécutée |

#### Dossier de sortie horodaté

Chaque batch crée un dossier dédié :
```
results/
  batch_20260127_143545/
    summary.json    # Rapport complet avec métriques
    tasks.csv       # Résultats par tâche (pour Excel/Python)
    README.txt      # Résumé rapide
```

#### Exécution non-bloquante

En mode batch, les visualisations sont **automatiquement désactivées** pour permettre une exécution sans interruption. Toutes les tâches sont traitées d'un coup, puis les résultats sont affichés à la fin.

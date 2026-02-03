# BRAIN Project - Capacités du Système

> **Dernière mise à jour :** Février 2026  
> **Version :** 2.5.0 (140 tasks, benchmark 3 modèles, fallbacks améliorés)

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

### Détection avancée de patterns (v1.7.0)

| Fonctionnalité | Description | Status |
|----------------|-------------|--------|
| **Patterns répétitifs** | Détecte si une grille est composée d'un motif qui se répète (tuile/pavage) | ✅ |
| **Sous-grilles** | Détecte les subdivisions rectangulaires régulières dans une grille | ✅ |
| **Objets avec contour** | Détecte les formes avec un intérieur d'une couleur et une bordure d'une autre | ✅ |

#### Exemple : Détection de pattern répétitif
```python
detector = SymbolDetector()
pattern_info = detector.detect_repeating_pattern(grid)
# Retourne: {
#   "pattern": [[1,2],[2,1]],  # Le motif de base
#   "tile_height": 2, "tile_width": 2,
#   "repetitions_h": 4, "repetitions_v": 3,
#   "coverage": 1.0  # 100% de la grille est couverte
# }
```

#### Exemple : Détection de sous-grilles
```python
subgrids = detector.detect_subgrids(grid)
# Retourne une liste de sous-grilles avec leur position et contenu
```

#### Exemple : Détection d'objets bordés
```python
bordered = detector.detect_bordered_objects(grid)
# Retourne: [{
#   "inner_color": 1,
#   "border_color": 2,
#   "inner_pixels": {...},
#   "border_pixels": {...}
# }]
```

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
| `tiling` | Répétition d'un motif pour remplir une grille plus grande | ✅ |
| `composite` | Combinaison de transformations (rotate+translate, etc.) | ✅ |
| `add_border` | Ajouter un contour coloré à un objet solide | ✅ |
| `flood_fill` | Remplissage de régions fermées avec une couleur | ✅ **NEW v2.4** |
| `symmetry` | Création de copies symétriques d'objets | ✅ **NEW v2.4** |
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
| `tile` | `repetitions_horizontal`, `repetitions_vertical` | Répéter un motif pour créer une grille plus grande | ✅ |
| `composite` | `transformations` (liste d'actions) | Combiner plusieurs transformations (rotate + translate, etc.) | ✅ |
| `add_border` | `border_color`, `color_filter` | Ajouter un contour coloré à un objet | ✅ |
| `flood_fill` | `seed_point`, `fill_color`, `connectivity` | Remplir une région connectée | ✅ **NEW v2.4** |
| `symmetry` | `axis`, `position`, `color_filter` | Créer une copie symétrique | ✅ **NEW v2.4** |

### Détails des axes de réflexion

| Axe | Description |
|-----|-------------|
| `horizontal` | Miroir haut-bas (flipud) |
| `vertical` | Miroir gauche-droite (fliplr) |
| `diagonal_main` | Miroir diagonale principale |
| `diagonal_anti` | Miroir anti-diagonale |

### Détails de l'action add_border (v1.10.0)

L'action `add_border` ajoute un contour coloré à un objet solide, en gardant l'intérieur avec sa couleur originale.

**Principe :**
- Les pixels de bordure (ayant au moins un voisin hors de l'objet) reçoivent la couleur du contour
- Les pixels intérieurs gardent la couleur originale

**Exemple JSON :**
```json
{
  "action": "add_border",
  "color_filter": 2,
  "params": {
    "border_color": 1
  }
}
```

**Exemple visuel :**
```
Input (3x3 red):    Output:
2 2 2               1 1 1
2 2 2      -->      1 2 1
2 2 2               1 1 1
```

**Cas supportés :**
- Carrés de toutes tailles (3x3, 4x4, 5x5, etc.)
- Rectangles
- Formes quelconques (blobs)

### Détails de l'action composite (v1.9.0)

L'action `composite` permet de combiner plusieurs transformations en séquence sur un même objet.

**Combinaisons supportées :**
- Rotation + Translation
- Réflexion + Translation
- Rotation + Changement de couleur
- Translation + Rotation + Changement de couleur
- etc.

**Exemple JSON :**
```json
{
  "action": "composite",
  "color_filter": 2,
  "params": {
    "transformations": [
      {"action": "rotate", "params": {"angle": 90}},
      {"action": "translate", "params": {"dx": 3, "dy": 1}}
    ]
  }
}
```

**Exemple avec changement de couleur :**
```json
{
  "action": "composite",
  "color_filter": 2,
  "params": {
    "transformations": [
      {"action": "reflect", "params": {"axis": "vertical"}},
      {"action": "translate", "params": {"dx": 2, "dy": -1}},
      {"action": "color_change", "params": {"from_color": 2, "to_color": 5}}
    ]
  }
}
```

**Ordre d'exécution :** Les transformations sont appliquées dans l'ordre de la liste. Le résultat de chaque transformation est utilisé comme entrée pour la suivante.

### Détails de l'action tile (v1.8.0)

L'action `tile` répète le pattern d'entrée pour créer une grille plus grande. Cette action est automatiquement détectée quand la grille de sortie est un multiple de la grille d'entrée.

**Détection automatique :**
- Le système détecte les changements de taille de grille **en priorité**
- Si `output_size = input_size × N`, vérifie si c'est un tiling parfait
- Calcule automatiquement `repetitions_horizontal` et `repetitions_vertical`

**Exemple JSON :**
```json
{
  "action": "tile",
  "params": {
    "repetitions_horizontal": 2,
    "repetitions_vertical": 2
  }
}
```

**Exemple : Input 2×2 → Output 4×4**
```
Input:        Output:
[1, 2]        [1, 2, 1, 2]
[2, 1]   →    [2, 1, 2, 1]
              [1, 2, 1, 2]
              [2, 1, 2, 1]
```

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

## 📁 Dataset de test (v2.5.0)

Le projet inclut **140 tâches de test** (10 par type de transformation) couvrant toutes les transformations supportées, avec une répartition équilibrée pour des analyses statistiques robustes.

### Répartition par type de transformation

| Type | Nombre | Fichiers |
|------|--------|----------|
| **Translation** | 10 | `task_translation_01` à `08`, `task_blob_translation`, `task_l_shape` |
| **Rotation** | 8 | `task_rotation_01` à `06`, `task_rotation_90`, `task_blob_rotation` |
| **Reflection** | 7 | `task_reflection_01` à `06`, `task_blob_reflection` |
| **Color change** | 7 | `task_color_change_01` à `06`, `task_blob_color_change` |
| **Draw line** | 5 | `task_draw_line_01` à `05` |
| **Add border** | 4 | `task_add_border_01` à `04` |
| **Tiling** | 5 | `task_tiling_01` à `03`, `task_pattern_tile_01`, `task_pattern_tile_02` |
| **Composite** | 4 | `task_composite_01` à `04` |
| **Flood fill** | 4 | `task_flood_fill_01` à `04` **(NEW v2.4)** |
| **Symmetry** | 4 | `task_symmetry_01` à `04` **(NEW v2.4)** |
| **Scale** | 4 | `task_scale_01` à `04` **(NEW v2.4)** |
| **Blob** | 4 | `task_blob_01` à `04` |
| **Multi-transform** | 3 | `task_multi_objects_01` à `03` |

### Variété des tests

Chaque type de transformation inclut des variations :

- **Formes différentes** : carrés, rectangles, L-shapes, T-shapes, blobs
- **Couleurs variées** : toutes les couleurs ARC (1-9)
- **Positions diverses** : coins, centre, bords
- **Paramètres variés** : dx/dy, angles, axes de réflexion
- **Tailles de grilles** : 6×6 à 9×9

### Utilisation

```bash
# Tester une seule tâche
python main.py --task data/task_translation_01.json

# Batch complet (52 tâches)
python main.py --batch data/

# Filtrer par type
python main.py --batch data/ --pattern "task_rotation_*.json"
python main.py --batch data/ --pattern "task_color_change_*.json"
```

---

## 📝 Historique des versions

### v2.5.0 (Février 2026) - Dataset 140 tâches + Benchmark 3 modèles
- ✅ **Dataset élargi** - 140 tâches (10 par type de transformation)
- ✅ **Benchmark complet** - Comparaison llama3, mistral, phi3 sur 140 tâches
- ✅ **Mistral recommandé** - 100/140 correct (71.4%), ~2x plus rapide que llama3
- ✅ **Fallbacks améliorés** - Direct fallback pour rotation/reflection (bypass LLM)
- ✅ **Composite executor** - Support color_change dans transformations composées
- ✅ **Auto-détection grid-level** - Rotation/reflection grid vs object-level
- ✅ **Script `generate_figures.py`** - Génération simplifiée des visualisations

### v2.4.0 (Février 2026) - Extended DSL + New Primitives
- ✅ **NOUVEAU: Action `flood_fill`** - Remplissage de régions fermées (enclosed regions, background)
- ✅ **NOUVEAU: Action `symmetry`** - Création de copies symétriques (vertical, horizontal, adjacent)
- ✅ **NOUVEAU: Action `scale`** - Mise à l'échelle d'objets (object-level scaling)
- ✅ **Détection automatique** - Les 3 nouvelles transformations sont détectées automatiquement
- ✅ **Direct fallback** - Exécution directe si confiance >= 0.85 (bypass LLM)
- ✅ **12 nouvelles tâches de test** - 4 par nouvelle primitive
- ✅ **DataLoader amélioré** - `load_latest_batch()` pour analyser uniquement le dernier batch
- ✅ **BatchRunner v1.11.0** - Rapport de couverture des transformations
- ✅ **64 tâches de test** au total

### v1.12.0 (Janvier 2026) - IEEE Publication Quality + Extended Dataset
- ✅ **NOUVEAU: Figures vectorielles PDF** - Sortie compatible LaTeX/Overleaf
- ✅ **Détection automatique de LaTeX** - Fallback gracieux avec DejaVu Serif
- ✅ **Tailles IEEE standardisées** - Single column (3.5in), double column (7.16in)
- ✅ **Palette colorblind-friendly** - Wong palette pour accessibilité
- ✅ **Fonts Computer Modern** - Compatibilité parfaite avec LaTeX
- ✅ **52 tâches de test** - Dataset élargi pour analyses statistiques
- ✅ **~10 tâches par transformation** - Répartition équilibrée

### v1.11.0 (Janvier 2026) - Data Analysis Module
- ✅ **NOUVEAU: Module `data_analysis/`** - Analyse des résultats de batch
- ✅ **DataLoader** - Charger et agréger les données de plusieurs batchs
- ✅ **MetricsCalculator** - Calculs statistiques (accuracy par transformation, t-tests, etc.)
- ✅ **AnalysisVisualizer** - Graphiques pour publications (barplots, boxplots, heatmaps)
- ✅ **ReportGenerator** - Export LaTeX, CSV, Markdown, JSON
- ✅ **Script `analyze.py`** - CLI pour analyse rapide
- ✅ **Données enrichies** - Timing breakdown, LLM vs fallback tracking, complexité

### v1.10.0 (Janvier 2026) - Add Border Action
- ✅ **NOUVEAU: Action `add_border`** - Ajouter un contour coloré à un objet
- ✅ **Détection automatique** - Le système détecte quand un objet reçoit un contour
- ✅ **Support de toutes les formes** - Carrés, rectangles, blobs
- ✅ Fichier de test: `task_add_border.json`

### v1.9.0 (Janvier 2026) - Composite Transformations
- ✅ **NOUVEAU: Action `composite`** - Combiner plusieurs transformations en séquence
- ✅ **Détection automatique** - Le système détecte rotation+translation, réflexion+translation, etc.
- ✅ **Exécution séquentielle** - Les transformations sont appliquées dans l'ordre
- ✅ **Support complet** - Rotation, réflexion, translation, changement de couleur
- ✅ Fichier de test: `task_composite_rotate_translate.json`

### v1.8.0 (Janvier 2026) - Grid Size Change Detection & Tiling
- ✅ **NOUVEAU: Détection de changement de taille de grille** - Le système priorise les transformations de taille différente
- ✅ **NOUVEAU: Action `tile`** - Répète un motif pour créer une grille plus grande
- ✅ **Détection précoce** - Les changements de taille sont vérifiés AVANT les autres transformations
- ✅ **Support de tiling** - Input 2×2 peut devenir Output 4×4 ou 6×6
- ✅ **Fallback intelligent** - Le système utilise les répétitions détectées si le LLM échoue
- ✅ Fichier de test: `task_pattern_tile.json`

### v1.7.0 (Janvier 2026) - Advanced Pattern Detection
- ✅ **Détection de patterns répétitifs** - `detect_repeating_pattern()` trouve le motif de base
- ✅ **Détection de sous-grilles** - `detect_subgrids()` trouve les subdivisions régulières
- ✅ **Détection d'objets bordés** - `detect_bordered_objects()` trouve les formes avec contour différent

### v1.6.0 (Janvier 2026) - Improved Prompting & Fallback
- ✅ **Prompt amélioré** - Few-shot examples concrets dans le system prompt
- ✅ **Instructions plus directes** - Le prompt génère le JSON exact à copier
- ✅ **Fallback automatique** - Si le LLM échoue, utilise les transformations détectées
- ✅ **Meilleure extraction des paramètres** - Parsing regex des transformations détectées
- ✅ **Auto-détection multi-transform** - Bascule automatique si différentes couleurs ont des transformations différentes
- ✅ **Réflexions grid-level vs object-level** - Distinction correcte entre les deux types
- ✅ **Draw line amélioré** - Meilleure détection et parsing du color
- ✅ Amélioration de la fiabilité globale du pipeline

### v1.5.0 (Janvier 2026) - Batch Evaluation Mode
- ✅ **NOUVEAU: Mode Batch** (`--batch DIR`) pour évaluer plusieurs tâches automatiquement
- ✅ **Module BatchRunner** - Exécute toutes les tâches et collecte des statistiques
- ✅ **Dossiers horodatés** - Chaque batch crée `results/batch_YYYYMMDD_HHMMSS/`
- ✅ **Rapports multiples** - `summary.json`, `tasks.csv`, `README.txt`
- ✅ **Exécution non-bloquante** - Visualisations désactivées pendant l'exécution
- ✅ **Navigateur interactif** - Parcourir les résultats avec boutons ◀/▶ et flèches clavier
- ✅ **Images sauvegardées** - `batch_summary.png` + images individuelles par tâche
- ✅ **Statistiques agrégées** - Accuracy moyenne, temps d'exécution, comptage des transformations
- ✅ Options: `--limit`, `--pattern`, `--output`, `--no-viz`

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

- [x] ~~Détection de patterns répétitifs~~ ✅ v1.7.0 / v1.8.0
- [x] ~~Détection de sous-grilles~~ ✅ v1.7.0
- [x] ~~Mode batch pour évaluer plusieurs tâches~~ ✅ v1.5.0
- [x] ~~Export des résultats en JSON~~ ✅ v1.5.0
- [x] ~~Taille de grille variable (tiling)~~ ✅ v1.8.0
- [x] ~~Support de transformations composées (translation + rotation simultanées)~~ ✅ v1.9.0
- [x] ~~Module d'analyse de données pour publications~~ ✅ v1.11.0
- [x] ~~Dataset élargi (~10 tâches par transformation)~~ ✅ v1.12.0
- [x] ~~Primitive `flood_fill` (remplissage régions fermées)~~ ✅ v2.4.0
- [x] ~~Primitive `symmetry` (création symétrie)~~ ✅ v2.4.0
- [x] ~~Primitive `scale` (mise à l'échelle objets)~~ ✅ v2.4.0
- [x] ~~Dataset 140 tâches (10 par transformation)~~ ✅ v2.5.0
- [x] ~~Benchmark 3 modèles (llama3, mistral, phi3)~~ ✅ v2.5.0
- [x] ~~Fallbacks améliorés (rotation, reflection)~~ ✅ v2.5.0
- [ ] Auto-détection du mode (single vs multi-transform)
- [ ] Détection de structures hiérarchiques (grilles dans grilles)
- [ ] Support de transformations conditionnelles (si couleur X alors...)

---

## ⚠️ Limitations connues

| Limitation | Description |
|------------|-------------|
| Couleurs différentes entre exemples | En mode standard, si chaque exemple a une couleur différente, utiliser `--multi` |
| Transformations composées | Une seule transformation par couleur en mode multi |
| ~~Taille de grille variable~~ | ✅ **Supporté depuis v1.8.0** (tiling) |
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

Chaque batch crée un dossier dédié avec visualisations :
```
results/
  batch_20260127_143545/
    summary.json           # Rapport complet avec métriques
    tasks.csv              # Résultats par tâche (pour Excel/Python)
    README.txt             # Résumé rapide
    images/
      batch_summary.png    # Vue d'ensemble de tous les tests
      task_xxx.png         # Image détaillée par tâche
```

#### Exécution non-bloquante

En mode batch, les visualisations sont **automatiquement désactivées pendant l'exécution** pour permettre un traitement sans interruption. À la fin du batch :
- Un **navigateur interactif** s'ouvre pour parcourir les résultats
- Les images sont **sauvegardées** dans le dossier `images/`

#### Navigateur interactif

À la fin du batch, une fenêtre interactive s'ouvre avec :
- **Input | Predicted | Expected | Difference** pour chaque tâche
- **Boutons ◀ Previous / Next ▶** pour naviguer
- **Flèches clavier** ← → pour navigation rapide
- **Touche Q** pour quitter
- **Statistiques** affichées en bas (n correct, accuracy moyenne)

Pour désactiver l'affichage final : `python main.py --batch data/ --no-viz`

---

## 📊 Module DATA_ANALYSIS (v1.12.0) - IEEE Publication Quality

Module d'analyse de données optimisé pour générer des **figures vectorielles PDF** compatibles avec **LaTeX/Overleaf** et les standards **IEEE**.

### Caractéristiques

- **Sortie vectorielle PDF** par défaut (qualité publication)
- **Détection automatique de LaTeX** (fallback gracieux si non installé)
- **Tailles IEEE standardisées** (single column: 3.5in, double column: 7.16in)
- **Palette colorblind-friendly** (Wong palette)
- **Fonts Computer Modern** (compatibles LaTeX)

### Structure

```
data_analysis/
├── __init__.py
├── data_loader.py      # Charger et agréger les résultats de batchs
├── metrics.py          # Calcul de métriques statistiques
├── visualizer.py       # Graphiques IEEE (matplotlib + LaTeX)
└── report_generator.py # Export LaTeX/CSV/Markdown
```

### Utilisation rapide

```bash
# Analyser tous les batchs (PDF vectoriel par défaut)
python analyze.py

# Figures IEEE single column (3.5 inches)
python analyze.py --ieee-size single

# Figures IEEE double column (7.16 inches)
python analyze.py --ieee-size double

# Formats multiples (PDF + PNG)
python analyze.py --fig-format pdf,png

# Générer uniquement les tableaux LaTeX
python analyze.py --format latex

# Mode interactif (afficher les graphiques)
python analyze.py --interactive
```

### Utilisation en Python

```python
from data_analysis import DataLoader, MetricsCalculator, AnalysisVisualizer, ReportGenerator

# 1. Charger les données
loader = DataLoader()
df = loader.load_all_batches("results/")

# 2. Calculer les métriques
calc = MetricsCalculator(df)
print(calc.accuracy_by_transformation())
print(calc.llm_vs_fallback_comparison())

# 3. Créer des visualisations IEEE (PDF vectoriel)
viz = AnalysisVisualizer(df, style="publication")

# Figures avec taille IEEE
viz.plot_accuracy_by_transformation(
    ieee_size="double",                    # 7.16 inches width
    save_path="figures/accuracy",          # Sans extension
    save_formats=["pdf", "png"]            # Multi-format
)

# Générer tous les plots d'un coup
viz.generate_all_plots(
    output_dir="figures/",
    formats=["pdf"]
)

# 4. Générer des rapports
gen = ReportGenerator(df, calc)
gen.generate_latex_tables("latex/")
gen.generate_markdown_report("report.md")
gen.generate_csv_summary("summary.csv")
```

### Visualisations disponibles

| Graphique | Description | Taille recommandée |
|-----------|-------------|-------------------|
| `plot_accuracy_by_transformation()` | Barplot accuracy par type | double |
| `plot_model_comparison()` | Comparaison par modèle LLM | single |
| `plot_accuracy_boxplot()` | Distribution des accuracies | double |
| `plot_confusion_matrix()` | Détection vs exécution | single |
| `plot_timing_breakdown()` | Temps (détection, LLM, exécution) | double |
| `plot_llm_vs_fallback()` | LLM vs fallback | double |
| `plot_accuracy_by_complexity()` | Scatter accuracy vs complexité | single |

### Tailles IEEE

| Size | Width | Usage |
|------|-------|-------|
| `single` | 3.5 in (88.9 mm) | IEEE single column |
| `double` | 7.16 in (181.9 mm) | IEEE double column |
| `full` | 7.16 × 9 in | Full page figure |

### Exports disponibles

| Format | Fichier | Usage |
|--------|---------|-------|
| **PDF** | `*.pdf` | **Vectoriel pour LaTeX** (recommandé) |
| PNG | `*.png` | Raster 300 DPI pour prévisualisations |
| LaTeX | `*.tex` | Tableaux pour articles scientifiques |
| CSV | `summary.csv`, `full_data.csv` | Analyse Excel/Pandas |
| Markdown | `report.md` | Documentation |
| JSON | `summary.json` | API/Intégration |

### Données collectées par tâche (enrichies v1.11.0)

| Champ | Description |
|-------|-------------|
| `primary_transformation` | Type principal détecté |
| `transformation_confidence` | Confiance (0-1) |
| `was_fallback_used` | Si le fallback a été utilisé |
| `llm_proposed_action` | Action proposée par le LLM |
| `timing_detection` | Temps de détection (s) |
| `timing_llm_response` | Temps de réponse LLM (s) |
| `timing_action_execution` | Temps d'exécution (s) |
| `complexity_num_colors` | Nombre de couleurs |
| `complexity_num_objects` | Nombre d'objets |

---

## 🚀 ROADMAP TIER 1-3 (v2.0.0)

Cette section documente les améliorations implémentées selon la roadmap en 3 niveaux.

### TIER 1 : Robustesse & Engineering

#### 1.1 Structured Logging (`modules/logger.py`)

Système de logging structuré pour le suivi du pipeline.

```python
from modules import BRAINLogger, LogLevel

logger = BRAINLogger(verbose=True, log_file="brain.log")

# Log a step
logger.step(LogLevel.DETECTION, "Found 3 objects", count=3)

# Timed step (automatic duration tracking)
with logger.timed_step(LogLevel.LLM, "Querying model"):
    response = llm.query(prompt)

# Get metrics
logger.print_metrics_summary()
```

| Feature | Description |
|---------|-------------|
| `LogLevel` | Composants: PIPELINE, PERCEPTION, DETECTION, PROMPTING, LLM, EXECUTION, ANALYSIS |
| Timing automatique | Contexte `timed_step` mesure la durée |
| Performance Metrics | Collecte LLM calls, temps par composant |
| Multi-output | Console (couleurs), fichier, JSON |

#### 1.2 Defensive Error Handling (`modules/executor.py`)

Gestion d'erreurs robuste dans l'exécuteur.

| Helper | Description |
|--------|-------------|
| `_safe_int()` | Conversion int sécurisée (gère strings, floats, mots) |
| `_safe_float()` | Conversion float sécurisée |
| `_safe_color()` | Conversion couleur (noms → nombres) |
| `_validate_grid()` | Validation de grille (NaN, dtype, empty) |
| `_get_params()` | Extraction sécurisée des params |

```python
# Avant (fragile)
dx = int(params.get("dx", 0))  # Crash si dx="three"

# Après (robuste)
dx = self._safe_int(params.get("dx", 0), default=0, name="dx")
# ⚠ Warning: Invalid dx='three', using default=0
```

#### 1.3 Resilient JSON Parsing (`modules/llm_client.py`)

Parsing JSON multi-stratégie pour gérer les erreurs LLM.

| Stratégie | Description |
|-----------|-------------|
| 1. Code block | `\`\`\`json {...} \`\`\`` |
| 2. Generic block | `\`\`\` {...} \`\`\`` |
| 3. Standalone | `{"action": ...}` dans le texte |
| 4. Fuzzy extraction | Reconstruction à partir de fragments |

**Fuzzy extraction gère :**
- Trailing commas
- Single quotes → double quotes
- Unquoted keys
- Comments in JSON

---

### TIER 2 : DSL Étendu (Nouvelles Actions)

Trois nouvelles primitives géométriques.

#### 2.1 Symmetry (`symmetry`)

Création de copies symétriques d'objets.

```json
{
  "action": "symmetry",
  "params": {
    "axis": "vertical",
    "position": "adjacent",
    "keep_original": true
  },
  "color_filter": 2
}
```

| Paramètre | Options | Description |
|-----------|---------|-------------|
| `axis` | horizontal, vertical, both, diagonal | Axe de symétrie |
| `position` | adjacent, opposite, {offset_x, offset_y} | Placement de la copie |
| `keep_original` | true/false | Conserver l'original |

#### 2.2 Flood Fill (`flood_fill`)

Remplissage de régions connectées (paint bucket).

```json
{
  "action": "flood_fill",
  "params": {
    "seed_point": {"row": 5, "col": 5},
    "fill_color": 3,
    "connectivity": 4
  }
}
```

| Paramètre | Options | Description |
|-----------|---------|-------------|
| `seed_point` | dict, "enclosed_regions", "background" | Point de départ |
| `fill_color` | 0-9 | Couleur de remplissage |
| `connectivity` | 4, 8 | Connectivité (4 ou 8 voisins) |
| `boundary_colors` | [int] | Couleurs formant barrière |

#### 2.3 Conditional Color (`conditional_color`)

Changements de couleur basés sur des conditions spatiales.

```json
{
  "action": "conditional_color",
  "params": {
    "rules": [
      {"condition": "is_edge", "from_color": 2, "to_color": 1},
      {"condition": "has_neighbor_color_0", "to_color": 3}
    ]
  }
}
```

| Condition | Description |
|-----------|-------------|
| `has_neighbor_color_X` | A un voisin de couleur X |
| `no_neighbor_color_X` | N'a pas de voisin de couleur X |
| `is_corner` | Pixel au coin de la grille |
| `is_edge` | Pixel sur le bord de la grille |
| `neighbor_count_ge_N` | ≥ N voisins non-fond |
| `neighbor_count_le_N` | ≤ N voisins non-fond |
| `is_isolated` | Aucun voisin non-fond |

---

### TIER 3 : Features Neuro-Symboliques Avancées

#### 3.1 Rule Memory / RAG (`modules/rule_memory.py`)

Système de mémoire pour l'apprentissage few-shot.

```python
from modules import RuleMemory

memory = RuleMemory("rule_memory.json")

# Stocker une règle réussie
memory.store_rule(
    task=task,
    action_data={"action": "translate", "params": {"dx": 2}},
    success=True,
    accuracy=1.0
)

# Trouver des règles similaires
similar = memory.find_similar_rules(new_task, top_k=3)

# Formater pour prompt few-shot
few_shot_text = memory.format_for_prompt(similar)
```

| Feature | Description |
|---------|-------------|
| `TaskSignature` | Extraction de features (shape, colors, transforms) |
| Similarity search | Matching par features (sans embeddings) |
| Persistence | Sauvegarde JSON automatique |
| Few-shot formatting | Génère texte pour prompt LLM |

**Task Signature Features:**
- Grid shapes (input/output)
- Colors (input, output, added, removed)
- Object counts and types
- Detected transformations

#### 3.2 Self-Correction Loop

Boucle d'auto-correction avec feedback d'erreur.

```bash
python main.py --task data/task.json --self-correct --max-retries 2
```

**Architecture:**

```
┌─────────────────────────────────────────────┐
│           SELF-CORRECTION LOOP              │
├─────────────────────────────────────────────┤
│                                             │
│  1. Initial Attempt                         │
│     ├── Query LLM (with RAG examples)       │
│     ├── Execute action                      │
│     └── Analyze result                      │
│                                             │
│  2. If incorrect:                           │
│     ├── Extract error feedback              │
│     ├── Create correction prompt            │
│     │   - Accuracy achieved                 │
│     │   - Pixel errors                      │
│     │   - Color confusions                  │
│     └── Re-query LLM                        │
│                                             │
│  3. Repeat (max_retries times)              │
│                                             │
│  4. Store result in Rule Memory             │
│                                             │
└─────────────────────────────────────────────┘
```

**Correction Prompt includes:**
- Previous action that failed
- Accuracy achieved
- Error count and pattern
- Color confusion matrix
- Suggestions for correction

#### 3.3 Nouvelles Options CLI

| Option | Description |
|--------|-------------|
| `--self-correct` | Activer la boucle d'auto-correction |
| `--max-retries N` | Nombre max de tentatives (défaut: 2) |
| `--no-memory` | Désactiver Rule Memory (RAG) |
| `--memory-path FILE` | Chemin du fichier mémoire |

---

## 📊 Résumé des Actions Supportées (v2.5.0)

| Action | TIER | Description | Status |
|--------|------|-------------|--------|
| `translate` | - | Translation (dx, dy) | ✅ |
| `rotate` | - | Rotation (90°, 180°, 270°) | ✅ |
| `reflect` | - | Réflexion (horizontal, vertical, diagonal) | ✅ |
| `scale` | **2** | Mise à l'échelle (facteur) | ✅ **v2.4** |
| `color_change` | - | Changement de couleur | ✅ |
| `fill` | - | Remplissage simple | ✅ |
| `copy` | - | Copie avec offset | ✅ |
| `replace_color` | - | Remplacement de couleur | ✅ |
| `draw_line` | - | Tracer ligne (Bresenham) | ✅ |
| `tile` | - | Pavage/Tiling | ✅ |
| `add_border` | - | Ajout de contour | ✅ |
| `composite` | - | Transformations combinées | ✅ |
| **`symmetry`** | **2** | **Création de symétrie (vertical, horizontal, adjacent)** | ✅ **v2.4** |
| **`flood_fill`** | **2** | **Remplissage régions fermées** | ✅ **v2.4** |
| `conditional_color` | 2 | Couleur conditionnelle | ⏳ Planned |

---

## 🧪 Tests et Validation

Pour tester les nouvelles fonctionnalités:

```bash
# Test TIER 1 - Logging
python -c "from modules import BRAINLogger, LogLevel; l=BRAINLogger(); l.step(LogLevel.PIPELINE, 'Test')"

# Test TIER 2 - New actions
python main.py --task data/mock_task.json

# Test TIER 3 - Self-correction
python main.py --task data/mock_task.json --self-correct --max-retries 2

# Test TIER 3 - Rule Memory
python -c "from modules import RuleMemory; m=RuleMemory(); print(m.get_statistics())"
```

---

## 🔄 Module MODEL COMPARATOR (v2.3.0)

Outil pour comparer les performances de différents modèles LLM.

### Architecture unifiée (v2.3.0)

**Important :** `compare_models.py` utilise maintenant `main.py --batch` (via `BatchRunner`) pour chaque modèle, garantissant des résultats **100% cohérents** avec le pipeline principal.

```
compare_models.py
     │
     ├── Model 1: BatchRunner(model="llama3")  → results/llama3/
     ├── Model 2: BatchRunner(model="mistral") → results/mistral/
     └── Model N: BatchRunner(model="...")     → results/.../
                    │
                    └── Même code que main.py --batch
```

### Modèles recommandés

| Modèle | Description | Taille | Installation |
|--------|-------------|--------|--------------|
| `mistral` | **🏆 RECOMMANDÉ** - Meilleur score et plus rapide | 4.1 GB | `ollama pull mistral` |
| `llama3` | Meta Llama 3 8B - Bon généraliste | 4.7 GB | `ollama pull llama3` |
| `phi3` | Microsoft Phi-3 Mini - Petit mais capable | 2.2 GB | `ollama pull phi3` |
| `gemma2` | Google Gemma 2 9B - Bon raisonnement | 5.4 GB | `ollama pull gemma2` |
| `codellama` | Meta Code Llama - Optimisé code/logique | 3.8 GB | `ollama pull codellama` |
| `qwen2` | Alibaba Qwen 2 7B - Multilingue, bonne logique | 4.4 GB | `ollama pull qwen2` |
| `llama3.1` | Meta Llama 3.1 8B - Dernière version | 4.7 GB | `ollama pull llama3.1` |
| `deepseek-coder` | DeepSeek Coder 6.7B - Spécialisé code | 3.8 GB | `ollama pull deepseek-coder` |

### Benchmark officiel (v2.5.0 - 140 tâches)

| Modèle | Tâches Correctes | Accuracy | Temps Moyen | Fallback |
|--------|------------------|----------|-------------|----------|
| 🏆 **mistral** | **100/140 (71.4%)** | **97.0%** | **6.9s** | 13.6% |
| llama3 | 98/140 (70.0%) | 94.8% | 11.4s | 13.6% |
| phi3 | 91/140 (65.0%) | 93.1% | 9.3s | 15.0% |

**Conclusion :** Mistral offre le meilleur compromis performance/vitesse. Il est ~2x plus rapide que llama3 tout en ayant un meilleur taux de réussite.

### Utilisation CLI

```bash
# Lister les modèles recommandés
python compare_models.py --list-models

# Comparer 2 modèles sur 5 tâches
python compare_models.py --models llama3 mistral --limit 5

# Comparaison complète avec visualisations
python compare_models.py --models llama3 mistral phi3 --visualize

# Comparaison sur toutes les tâches
python compare_models.py --models llama3 mistral --output comparison_full/

# Générer uniquement les visualisations (depuis résultats existants)
python compare_models.py --viz-only comparison_results/
```

### Ce qui se passe en interne

Pour chaque modèle, `compare_models.py` :
1. Crée un `BatchRunner` avec ce modèle
2. Exécute `runner.run_batch()` (identique à `main.py --batch`)
3. Sauvegarde les résultats dans `output_dir/model_name/`
4. Agrège les résultats pour la comparaison

### Rapports générés

| Fichier | Format | Contenu |
|---------|--------|---------|
| `comparison.json` | JSON | Résultats complets avec détails |
| `model_summary.csv` | CSV | Résumé par modèle (accuracy, temps, etc.) |
| `detailed_results.csv` | CSV | Résultats par tâche et modèle |
| `comparison_report.md` | Markdown | Rapport formaté pour lecture |

### Métriques collectées

| Métrique | Description |
|----------|-------------|
| `accuracy` | Précision moyenne (0-1) |
| `correct_count` | Nombre de tâches résolues |
| `avg_response_time` | Temps de réponse moyen (ms) |
| `fallback_rate` | % d'utilisation du fallback |

### Visualisations de comparaison (v2.2.0)

7 types de graphiques générés automatiquement en PNG et PDF :

| Graphique | Description |
|-----------|-------------|
| `accuracy_comparison` | Barplot accuracy par modèle |
| `time_comparison` | Barplot temps de réponse par modèle |
| `accuracy_vs_time` | Scatter plot accuracy vs temps (trade-off) |
| `accuracy_boxplot` | Distribution des accuracies par modèle |
| `per_task_comparison` | Barplot groupé accuracy par tâche |
| `fallback_comparison` | Taux d'utilisation du fallback |
| `summary_dashboard` | Dashboard 2x2 avec toutes les métriques |

#### Commandes de visualisation

```bash
# Générer visualisations depuis résultats existants
python compare_models.py --viz-only comparison_results/

# Nouvelle comparaison AVEC visualisations
python compare_models.py --models llama3 mistral --limit 10 --visualize

# Comparaison complète avec graphiques
python compare_models.py -m llama3 mistral -v -o comparison_full/
```

#### Utilisation en Python

```python
from modules.model_comparator import ModelComparisonVisualizer

# Depuis résultats existants
viz = ModelComparisonVisualizer(results_path="comparison_results/comparison.json")

# Ou depuis un objet ModelComparisonResult
viz = ModelComparisonVisualizer(comparison=results)

# Générer un graphique spécifique
viz.plot_accuracy_comparison(save_path="accuracy.png", show=True)
viz.plot_summary_dashboard(save_path="dashboard.pdf")

# Générer tous les graphiques
viz.save_all_plots("output/figures/", formats=['png', 'pdf'])
```

### Installation rapide (3 modèles)

```bash
# Installer les modèles
ollama pull llama3
ollama pull mistral  
ollama pull phi3

# Lancer la comparaison
python compare_models.py -m llama3 mistral phi3 -l 10
```

# BRAIN Project - Capacités du Système

> **Dernière mise à jour :** Janvier 2026  
> **Version :** 1.0.0

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
| `blob` | Forme irrégulière/quelconque | ✅ |

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

---

## ⚡ Module EXECUTOR (Actions)

### Actions supportées

| Action | Paramètres | Description | Status |
|--------|------------|-------------|--------|
| `translate` | `dx`, `dy`, `color_filter` | Déplace les pixels | ✅ |
| `fill` | `color`, `region` | Remplit une zone | ✅ |
| `replace_color` | `from_color`, `to_color` | Change une couleur | ✅ |
| `copy` | `dx`, `dy`, `color_filter` | Copie avec offset | ✅ |
| `rotate` | `angle`, `color_filter` | Rotation | 🔜 À venir |
| `reflect` | `axis`, `color_filter` | Réflexion | 🔜 À venir |
| `scale` | `factor`, `color_filter` | Agrandir/réduire | 🔜 À venir |

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

- [ ] Actions `rotate`, `reflect`, `scale` dans l'executor
- [ ] Détection de patterns répétitifs
- [ ] Détection de sous-grilles
- [ ] Support multi-objets dans les transformations
- [ ] Mode batch pour évaluer plusieurs tâches
- [ ] Export des résultats en JSON

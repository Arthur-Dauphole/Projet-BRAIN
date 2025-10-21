#  Feuille de Route du Projet de Détection Géométrique 🚀

Ce document liste les prochaines étapes et les idées d'amélioration pour faire évoluer le projet. Cochez les cases (`- [x]`) au fur et à mesure de la progression.

---

## Étape 1 : Enrichir la détection de formes

- [ ] **Implémenter un `CircleDetector`**
  - *Logique :* Vérifier si les pixels d'une forme sont à une distance quasi constante d'un point central (le centre de la `BoundingBox` est un bon début).
  - *Pistes :* Calculer la distance moyenne et l'écart-type de tous les pixels au centre. Si l'écart-type est faible, c'est probablement un cercle.

- [ ] **Implémenter un `TriangleDetector`**
  - *Logique :* Identifier des formes possédant 3 points "extrêmes" ou coins.
  - *Pistes :* Utiliser un algorithme de détection de coins ou trouver l'enveloppe convexe (`convex hull`) de la forme et voir si elle a 3 sommets.

- [ ] **Implémenter un détecteur de "Blobs" (formes non-géométriques)**
  - *Logique :* Classifier les formes restantes qui ne sont ni des lignes, ni des rectangles, etc.
  - *Propriétés à calculer :* Le "moment" (centre de masse) de la forme, ou sa "squelettisation" pour en comprendre la structure (avec `skimage.morphology.skeletonize` par exemple).

---

## Étape 2 : Analyser les relations entre les formes

- [ ] **Détection de l'inclusion (forme dans une forme)**
  - *Logique :* Ajouter une fonction qui vérifie si la `BoundingBox` d'une forme est entièrement contenue dans une autre.
  - *Pour plus de précision :* Vérifier ensuite que tous les pixels de la forme interne sont bien dans l'ensemble des pixels de la forme externe.

- [ ] **Analyse des relations spatiales (gauche de, au-dessus, etc.)**
  - *Logique :* Comparer les coordonnées des `BoundingBox` pour déterminer les positions relatives.
  - *Exemple :* `shape_A.bbox.max_x < shape_B.bbox.min_x` signifie que A est entièrement à gauche de B.

- [ ] **Reconnaissance de motifs simples**
  - *Logique :* Après avoir détecté toutes les formes et leurs relations, chercher des séquences.
  - *Exemple :* Trouver tous les groupes de 3 carrés de même couleur qui sont alignés horizontalement.

---

## Étape 3 : Optimiser et professionnaliser

- [ ] **Optimisation des performances**
  - *Objectif :* Remplacer la fonction `extract_connected_components` maison par une méthode beaucoup plus rapide pour les grandes grilles.
  - *Solution :* Utiliser la fonction `scipy.ndimage.label` qui est écrite en C et est extrêmement performante.

- [ ] **Migration vers un framework de test standard**
  - *Objectif :* Rendre les tests plus modulaires, plus puissants et plus faciles à écrire.
  - *Solution :* Adapter `test_runner.py` pour qu'il utilise le framework **pytest**. Cela te permettra de simplement écrire des fonctions de test `test_quelquechose()` sans toute la structure de classe `TestSuite`.
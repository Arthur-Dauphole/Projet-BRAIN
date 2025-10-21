import json

def parcourir_grille(grille):
    
    cases_colorees = {}
    for x in range(len(grille)):
        for y in range(len(grille[x])):
            couleur = grille[x][y]
            if couleur != 0:
                if couleur not in cases_colorees:
                    cases_colorees[couleur] = []
                cases_colorees[couleur].append((x, y))

    return cases_colorees

#fonction pour voir si y'a des lignes (selon les cordonnées des cases colorées)
def detecter_ligne(cases_colorees):
    lignes_detectees = []
    for couleur, coordonnees in cases_colorees.items():
        if len(coordonnees) < 2:
            continue

        coordonnees_triees = sorted(coordonnees)

        ligne_horizontale = verifier_ligne_horizontale(coordonnees_triees)
        if ligne_horizontale:
            lignes_detectees.append({'type': 'horizontale','couleur': couleur,'debut': ligne_horizontale[0],'fin': ligne_horizontale[1]})
            continue

        ligne_verticale = verifier_ligne_verticale(coordonnees_triees)
        if ligne_verticale:
            lignes_detectees.append({'type': 'verticale','couleur': couleur,'debut': ligne_verticale[0],'fin': ligne_verticale[1]})
            continue

        ligne_diagonale = verifier_ligne_diagonale(coordonnees_triees)
        if ligne_diagonale:
            lignes_detectees.append({
                'type': ligne_diagonale[2],  # type de diagonale
                'couleur': couleur,
                'debut': ligne_diagonale[0],
                'fin': ligne_diagonale[1]
            })
    return lignes_detectees

#fonction pour voir si la ligne est horizontale
def verifier_ligne_horizontale(coordonnees):
    par_x = {}
    for x, y in coordonnees:
        if x not in par_x:
            par_x[x] = []
        par_x[x].append(y)
    
    for x, liste_y in par_x.items():
        if len(liste_y) >= 2:
            liste_y_triee = sorted(liste_y)
            
            
            est_continue = True
            for i in range(len(liste_y_triee) - 1):
                if liste_y_triee[i + 1] - liste_y_triee[i] != 1:
                    est_continue = False
                    break
            
            if est_continue:
                return ((x, liste_y_triee[0]), (x, liste_y_triee[-1]))
    
    return None

def verifier_ligne_verticale(coordonnees):
    par_y = {}
    for x, y in coordonnees:
        if y not in par_y:
            par_y[y] = []
        par_y[y].append(x)
    
    for y, liste_x in par_y.items():
        if len(liste_x) >= 2:
            liste_x_triee = sorted(liste_x)
            
            est_continue = True
            for i in range(len(liste_x_triee) - 1):
                if liste_x_triee[i + 1] - liste_x_triee[i] != 1:
                    est_continue = False
                    break
            
            if est_continue:
                return ((liste_x_triee[0], y), (liste_x_triee[-1], y))
    
    return None


def verifier_ligne_diagonale(coordonnees):
    if len(coordonnees) < 2:
        return None
    
    coordonnees_triees = sorted(coordonnees)
    
    est_diag_desc = True
    for i in range(len(coordonnees_triees) - 1):
        x1, y1 = coordonnees_triees[i]
        x2, y2 = coordonnees_triees[i + 1]
        
        if x2 - x1 != 1 or y2 - y1 != 1:
            est_diag_desc = False
            break
    
    if est_diag_desc:
        return (coordonnees_triees[0], coordonnees_triees[-1], 'diagonale_↘')

    est_diag_mont = True
    for i in range(len(coordonnees_triees) - 1):
        x1, y1 = coordonnees_triees[i]
        x2, y2 = coordonnees_triees[i + 1]
        
        if x2 - x1 != 1 or y2 - y1 != -1:
            est_diag_mont = False
            break
    
    if est_diag_mont:
        return (coordonnees_triees[0], coordonnees_triees[-1], 'diagonale_↗')
    
    return None

def afficher_resultats(lignes_detectees):
    """Affiche les résultats de manière lisible."""
    if not lignes_detectees:
        print("Aucune ligne détectée")
        return
    
    print(f"✅ {len(lignes_detectees)} ligne(s) détectée(s) :\n")
    
    for i, ligne in enumerate(lignes_detectees, 1):
        print(f"Ligne {i}:")
        print(f"  • Type: {ligne['type']}")
        print(f"  • Couleur: {ligne['couleur']}")
        print(f"  • De {ligne['debut']} à {ligne['fin']}")
        print()

with open('/Users/paullefrais/Documents/ISAE SUPAERO/Cours Supaero/2A/Projet R&D Brain/Projet-BRAIN-VSCODE/Fichiers Json test/test_diagonale.json', 'r') as f:
    grilles = json.load(f)  # Charger toutes les grilles

# Tester chaque grille du fichier
for i, grille in enumerate(grilles, 1):
    print(f"\n{'='*50}")
    print(f"GRILLE {i}")
    print(f"{'='*50}")
    
    cases = parcourir_grille(grille)
    lignes = detecter_ligne(cases)
    afficher_resultats(lignes)
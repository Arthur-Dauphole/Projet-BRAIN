"""
Utilitaires de couleur pour les grilles ARC-AGI.

Fournit un mappage stable entre codes de couleur entiers et noms lisibles,
partagé entre perception, raisonnement et pont LLM.
"""

from __future__ import annotations

from typing import Dict


class ColorMapper:
    """Mappe les codes de couleur numériques vers des noms lisibles."""

    # Palette ARC standard (0-9)
    COLOR_NAMES: Dict[int, str] = {
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Grey",
        6: "Magenta",
        7: "Orange",
        8: "Cyan",
        9: "Brown",
    }

    @classmethod
    def get_color_name(cls, color_code: int) -> str:
        """
        Retourne le nom lisible associé au code couleur.

        Args:
            color_code: Code de couleur entier (0-9 typiquement).

        Returns:
            Nom de couleur lisible (par ex. 'Red'), ou 'ColorX' en fallback.
        """
        return cls.COLOR_NAMES.get(color_code, f"Color{color_code}")


__all__ = ["ColorMapper"]


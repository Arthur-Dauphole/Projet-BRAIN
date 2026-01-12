"""
Color abstraction utilities for ARC-style grids.

This module provides a stable mapping between integer color codes
and human-readable color names, shared across perception and reasoning.
"""
from matplotlib.colors import ListedColormap

class ColorMapper:
    """Maps numerical color codes to readable symbolic names."""

    # Symbolic names for debug/logging
    COLOR_NAMES = {
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
    
    # Official ARC-AGI hex colors for visualization
    COLOR_HEX = [
        '#000000',  # 0 Black
        '#0074D9',  # 1 Blue
        '#FF4136',  # 2 Red
        '#2ECC40',  # 3 Green
        '#FFDC00',  # 4 Yellow
        '#AAAAAA',  # 5 Grey
        '#F012BE',  # 6 Magenta
        '#FF851B',  # 7 Orange
        '#7FDBFF',  # 8 Cyan
        '#870C25',  # 9 Brown
    ]
    
    # Matplotlib colormap for plotting
    ARC_CMAP = ListedColormap(COLOR_HEX)

    @classmethod
    def name(cls, code):
        """Return the symbolic color name associated with a numeric code."""
        return cls.COLOR_NAMES.get(code, f"Color{code}")
    
    @classmethod
    def hex(cls, code: int) -> str:
        """Return the hex color string for a numeric code (0-9)."""
        if 0 <= code < len(cls.COLOR_HEX):
            return cls.COLOR_HEX[code]
        return "#000000"  # fallback black
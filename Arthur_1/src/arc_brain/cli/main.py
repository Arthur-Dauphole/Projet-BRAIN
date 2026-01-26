"""
CLI principale pour le projet ARC-BRAIN.

Fourni une commande `solve` qui orchestrera, à terme, perception → raisonnement → bridge LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from arc_geometric_detection import GeometricDetectionEngine
from arc_neuro_bridge import PromptFactory, SceneDescriber


def _load_grid_from_json(path: Path) -> Dict[str, Any]:
    """Charge un fichier JSON de tâche ARC-like (format proche de arc_test_grids)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def cmd_solve(args: argparse.Namespace) -> None:
    """
    Commande `solve` : pour l’instant, pipeline démo perception → description → prompt.
    """
    json_path = Path(args.task)
    data = _load_grid_from_json(json_path)

    # Pour une première version, on prend le premier test_grids comme exemple.
    test_case = data["test_grids"][0]
    grid = np.array(test_case["grid"])

    print(f"Loaded task '{test_case['id']}' from {json_path}")
    print(f"Grid shape: {grid.shape[::-1]}")

    engine = GeometricDetectionEngine(background_color=0)
    analysis = engine.analyze_grid(grid, verbose=False)

    shapes = []
    for lst in analysis["detected_shapes"].values():
        shapes.extend(lst)

    describer = SceneDescriber(include_spatial_relations=True)
    scene_description = describer.describe_scene(shapes)

    prompt = PromptFactory.create_prompt(
        input_scene=scene_description,
        output_scene="(Aucune sortie fournie dans cette démo CLI)",
        task_description="Demo CLI arc-brain: perception + description sans résolution complète.",
    )

    print("\n=== SCENE DESCRIPTION ===")
    print(scene_description)
    print("\n=== GENERATED PROMPT ===")
    print(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(prog="arc-brain", description="ARC-BRAIN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser(
        "solve",
        help="Résoudre (ou analyser) une tâche ARC-like à partir d’un JSON.",
    )
    solve_parser.add_argument(
        "task",
        type=str,
        help="Chemin vers un fichier JSON de tâche (format proche de arc_test_grids.json).",
    )
    solve_parser.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


"""
data_loader.py - Chargement et agrégation des données de batch
==============================================================

Charge les résultats de plusieurs batchs et les combine en un DataFrame
pour l'analyse statistique.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


class DataLoader:
    """
    Charge et agrège les résultats de batch pour l'analyse.
    
    Example:
        loader = DataLoader()
        df = loader.load_all_batches("results/")
        print(df.columns)
    """
    
    def __init__(self):
        """Initialise le DataLoader."""
        self.batch_metadata: List[Dict[str, Any]] = []
    
    def load_batch(self, batch_folder: str) -> pd.DataFrame:
        """
        Charge un seul batch depuis son dossier.
        
        Args:
            batch_folder: Chemin vers le dossier du batch
            
        Returns:
            DataFrame avec les résultats du batch
        """
        folder = Path(batch_folder)
        
        # Chercher le fichier summary.json
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found in {batch_folder}")
        
        with open(summary_path, 'r') as f:
            data = json.load(f)
        
        # Extraire les métadonnées du batch
        metadata = data.get("metadata", {})
        batch_info = {
            "batch_id": folder.name,
            "model": metadata.get("model_used", "unknown"),
            "timestamp": metadata.get("run_timestamp", ""),
            "version": metadata.get("program_version", "unknown"),
        }
        self.batch_metadata.append(batch_info)
        
        # Convertir les résultats des tâches en DataFrame
        task_results = data.get("task_results", [])
        
        if not task_results:
            return pd.DataFrame()
        
        # Créer le DataFrame
        df = pd.DataFrame(task_results)
        
        # Ajouter les métadonnées du batch à chaque ligne
        df["batch_id"] = folder.name
        df["model"] = metadata.get("model_used", "unknown")
        df["batch_timestamp"] = metadata.get("run_timestamp", "")
        df["program_version"] = metadata.get("program_version", "unknown")
        
        # Extraire les sous-dictionnaires (timing, complexity)
        if "timing" in df.columns:
            timing_df = pd.json_normalize(df["timing"])
            timing_df.columns = [f"timing_{col}" for col in timing_df.columns]
            df = pd.concat([df.drop("timing", axis=1), timing_df], axis=1)
        
        if "complexity" in df.columns:
            complexity_df = pd.json_normalize(df["complexity"])
            complexity_df.columns = [f"complexity_{col}" for col in complexity_df.columns]
            df = pd.concat([df.drop("complexity", axis=1), complexity_df], axis=1)
        
        return df
    
    def load_latest_batch(
        self, 
        results_dir: str = "results/",
        pattern: str = "batch_*"
    ) -> pd.DataFrame:
        """
        Charge uniquement le batch le plus récent.
        
        Args:
            results_dir: Répertoire contenant les dossiers de batch
            pattern: Pattern glob pour les dossiers de batch
            
        Returns:
            DataFrame avec les résultats du dernier batch uniquement
        """
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Trouver tous les dossiers de batch et les trier par nom (qui contient le timestamp)
        batch_folders = sorted(results_path.glob(pattern))
        
        if not batch_folders:
            print(f"No batch folders found matching '{pattern}' in {results_dir}")
            return pd.DataFrame()
        
        # Prendre le dernier (le plus récent par ordre alphabétique du timestamp)
        latest_folder = batch_folders[-1]
        
        print(f"Found {len(batch_folders)} batch folders")
        print(f"📌 Loading LATEST batch only: {latest_folder.name}")
        
        try:
            df = self.load_batch(str(latest_folder))
            df = self._convert_types(df)
            print(f"✓ Loaded {len(df)} task results from {latest_folder.name}")
            return df
        except Exception as e:
            print(f"✗ Failed to load {latest_folder.name}: {e}")
            return pd.DataFrame()
    
    def load_all_batches(
        self, 
        results_dir: str = "results/",
        pattern: str = "batch_*"
    ) -> pd.DataFrame:
        """
        Charge tous les batchs d'un répertoire.
        
        Args:
            results_dir: Répertoire contenant les dossiers de batch
            pattern: Pattern glob pour les dossiers de batch
            
        Returns:
            DataFrame combiné de tous les batchs
        """
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Trouver tous les dossiers de batch
        batch_folders = sorted(results_path.glob(pattern))
        
        if not batch_folders:
            print(f"No batch folders found matching '{pattern}' in {results_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(batch_folders)} batch folders")
        
        # Charger chaque batch
        all_dfs = []
        for folder in batch_folders:
            try:
                df = self.load_batch(str(folder))
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  ✓ Loaded {folder.name}: {len(df)} tasks")
            except Exception as e:
                print(f"  ✗ Failed to load {folder.name}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Combiner tous les DataFrames
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Convertir les types
        combined = self._convert_types(combined)
        
        print(f"\nTotal: {len(combined)} task results loaded")
        return combined
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Charge directement depuis un fichier CSV.
        
        Args:
            csv_path: Chemin vers le fichier CSV
            
        Returns:
            DataFrame avec les données
        """
        df = pd.read_csv(csv_path)
        return self._convert_types(df)
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les colonnes aux bons types."""
        # Colonnes booléennes
        bool_cols = ['success', 'is_correct', 'was_fallback_used']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Colonnes numériques
        num_cols = ['accuracy', 'execution_time', 'transformation_confidence',
                    'timing_total', 'timing_llm_response', 'timing_detection', 
                    'timing_action_execution', 'complexity_num_colors', 
                    'complexity_num_objects']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_batch_summary(self) -> pd.DataFrame:
        """
        Retourne un résumé de tous les batchs chargés.
        
        Returns:
            DataFrame avec les métadonnées de chaque batch
        """
        return pd.DataFrame(self.batch_metadata)
    
    def filter_by_model(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """Filtre le DataFrame par modèle."""
        return df[df["model"] == model]
    
    def filter_by_transformation(
        self, 
        df: pd.DataFrame, 
        transformation: str
    ) -> pd.DataFrame:
        """Filtre le DataFrame par type de transformation."""
        return df[df["primary_transformation"] == transformation]
    
    def filter_by_date_range(
        self, 
        df: pd.DataFrame, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Filtre le DataFrame par plage de dates.
        
        Args:
            df: DataFrame à filtrer
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
        """
        if "batch_timestamp" not in df.columns:
            return df
        
        df["_date"] = pd.to_datetime(df["batch_timestamp"], errors='coerce')
        mask = (df["_date"] >= start_date) & (df["_date"] <= end_date)
        result = df[mask].drop("_date", axis=1)
        return result

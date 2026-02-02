import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import des fonctions existantes
from exam_runner import run_exam, show_exam_report

HISTORY_FILE = "exam_history.json"
BATCH_FILE = "current_batch.json"

RULE_COLORS = {
    "Translation": "#3498db", 
    "Rotation": "#9b59b6",    
    "Symmetry": "#e67e22",    
    "Color": "#e74c3c",       
    "Fill": "#2ecc71",        
    "Unknown": "#95a5a6"
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list): return {"processed_batches": [], "results": data}
                return data
        except: pass
    return {"processed_batches": [], "results": []}

def save_history(data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def update_history_safe(new_results):
    current_batch_id = None
    if os.path.exists(BATCH_FILE):
        with open(BATCH_FILE, 'r') as f:
            current_batch_id = json.load(f).get("batch_id")
    
    history_data = load_history()
    
    if current_batch_id and current_batch_id in history_data["processed_batches"]:
        print(f"\n[Dashboard] Batch {current_batch_id[-6:]} déjà existant.")
        return history_data["results"]
    
    print(f"\n[Dashboard] Ajout du Batch {current_batch_id[-6:]}...")
    
    timestamp = float(current_batch_id) if current_batch_id else 0
    
    for res in new_results:
        detected_raw = res['detected'].lower()
        detected_rule = "None"
        for kw in ["translation", "rotation", "symmetry", "color", "fill"]:
            if kw in detected_raw:
                detected_rule = kw.capitalize()
                break
                
        entry = {
            "batch_id": current_batch_id,
            "timestamp": timestamp,
            "expected": res['expected'],
            "detected_guess": detected_rule,
            "success": res['success']
        }
        history_data["results"].append(entry)
    
    if current_batch_id:
        history_data["processed_batches"].append(current_batch_id)
        
    save_history(history_data)
    return history_data["results"]

def plot_dashboard_v2(history):
    if not history: return

    df = pd.DataFrame(history)
    
    # --- CONFIGURATION LAYOUT AÉRÉ ---
    # Taille augmentée pour le plein écran
    fig = plt.figure(figsize=(24, 13)) 
    
    # GridSpec avec marges forcées (hspace/wspace) et ratios ajustés
    # height_ratios=[1, 1.2] donne plus de place à la matrice en bas
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.5, wspace=0.25)
    
    # Fond légèrement gris pour faire ressortir les graphs
    fig.patch.set_facecolor('#fdfdfd')

    # --- GRAPH 1 : TAUX DE RÉUSSITE PAR RÈGLE ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    stats = df.groupby('expected')['success'].mean() * 100
    colors = [RULE_COLORS.get(idx, "#333") for idx in stats.index]
    bars = ax1.bar(stats.index, stats.values, color=colors, alpha=0.85, edgecolor='#333', linewidth=1)
    
    ax1.set_title("1. Success Rate by Rule", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
    ax1.set_ylabel("Success (%)", fontsize=12)
    ax1.set_ylim(0, 115) # Un peu plus haut pour laisser de la place au texte
    ax1.grid(axis='y', alpha=0.6, linestyle='--')
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1)
    
    # Annotations plus grosses et aérées
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f"{height:.0f}%", 
                 ha='center', va='bottom', fontweight='bold', fontsize=12, color='#34495e')

    # --- GRAPH 2 : ÉVOLUTION TEMPORELLE ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'timestamp' in df.columns:
        evolution = df.groupby('timestamp')['success'].mean() * 100
        sessions = [f"S{i+1}" for i in range(len(evolution))]
        
        ax2.plot(sessions, evolution.values, marker='o', markersize=8, linestyle='-', linewidth=3, color='#2c3e50')
        ax2.fill_between(sessions, evolution.values, alpha=0.1, color='#2c3e50')
        
        ax2.set_title("2. Progression Globale (Timeline)", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
        ax2.set_ylabel("Précision Globale (%)", fontsize=12)
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Afficher la valeur du dernier point
        last_val = evolution.values[-1]
        ax2.text(sessions[-1], last_val + 3, f"{last_val:.0f}%", ha='center', fontweight='bold', color='#e74c3c', fontsize=12)
    else:
        ax2.text(0.5, 0.5, "Données insuffisantes", ha='center')

    # --- GRAPH 3 : MATRICE DE CONFUSION ---
    ax3 = fig.add_subplot(gs[1, :]) 
    
    labels = ["Translation", "Rotation", "Symmetry", "Color", "Fill", "None"]
    
    confusion = pd.crosstab(df['expected'], df['detected_guess'])
    confusion = confusion.reindex(index=labels[:-1], columns=labels, fill_value=0)
    
    # Heatmap avec cases carrées et annotations lisibles
    sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", ax=ax3, 
                cbar=False, annot_kws={"size": 12, "weight": "bold"},
                linewidths=1, linecolor='white') # Lignes blanches entre les cases
    
    ax3.set_title("3. Confusion Matrix (Error Diagnosis)", fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax3.set_ylabel("Expected Rule (generator)", fontsize=11)
    ax3.set_xlabel("AI detected rule", fontsize=11)
    
    # Rotation des labels pour éviter le chevauchement
    ax3.tick_params(axis='x', rotation=0, labelsize=11)
    ax3.tick_params(axis='y', rotation=0, labelsize=11)

    # Titre Global du Dashboard
    plt.suptitle("AI Performance Dashboard (BRAIN Project)", fontsize=22, fontweight='bold', y=0.99)
    
    # --- MODIFICATION ICI ---
    # J'ai passé le "left" de 0.05 à 0.15 pour laisser la place aux textes à gauche
    plt.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.08)
    
    plt.show()

if __name__ == "__main__":
    # 1. Run
    results, accuracy = run_exam()
    # 2. Update
    all_results = update_history_safe(results)
    # 3. Report (Fermer pour voir le dashboard)
    show_exam_report(results, accuracy)
    # 4. Dashboard V2
    plot_dashboard_v2(all_results)
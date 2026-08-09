#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def load_stage_data(folder_path, stage_keyword, tolerance="0.000050"):
    """
    Sucht nach Dateien für die jeweilige Stage. 
    Bei PCG wird die Toleranz im Namen erwartet, bei QDLDL nicht.
    """
    if "qdldl" in folder_path.name.lower():
        search_pattern = f"*{stage_keyword}*.result"
    else:
        search_pattern = f"*{tolerance}*{stage_keyword}*.result"
        
    files = list(folder_path.glob(search_pattern))
    
    data = []
    if not files:
        return np.array(data)
        
    for file_path in files:
        with open(file_path, 'r') as file:
            for line in file:
                val = line.strip().replace(',', '')
                if val:
                    try:
                        data.append(float(val))
                    except ValueError:
                        pass
                        
    return np.array(data)


def plot_gantt_chart(folder_path):
    stages = ["ktt_times", "shur_times", "linsys_times", "dz_times", "line_search_times"]
    stage_names = ["KKT", "Schur", "Linsys", "DZ", "Line Search"]

    tolerance = "0.000050"
    colors = ['#ADD8E6', '#90EE90', '#F08080', '#FFFACD', '#87CEFA'] 
    
    means = []
    stds = []
    
    print(f"\n--- Analysiere Ordner: {folder_path.name} ---")
    
    # 1. Daten laden und Statistiken berechnen
    for stage in stages:
        data = load_stage_data(folder_path, stage, tolerance)
        
        if len(data) > 0:
            means.append(np.mean(data))
            stds.append(np.std(data))
            # HIER ist die neue Log-Ausgabe für die genutzten Zeilen
            print(f"  [+] {len(data):>5} Zeilen für '{stage}' verwendet.")
        else:
            print(f"  [!] Warnung: Keine Daten für '{stage}' in {folder_path.name} gefunden.")
            means.append(0.0)
            stds.append(0.0)
            
    # Falls der Ordner komplett leer ist, abbrechen
    if sum(means) == 0:
        print(f"[!] Überspringe {folder_path.name} (Keine passenden Timing-Daten gefunden).")
        return
        
    # 2. Den Plot vorbereiten
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    y_pos = np.arange(len(stage_names))[::-1] 
    
    starts = [0]
    for i in range(len(means)-1):
        starts.append(starts[-1] + means[i])
        
    y_labels = [f"{name} ({m:.2f} ± {s:.2f})" for name, m, s in zip(stage_names, means, stds)]
    
    # 3. Zeichnen der Elemente
    for i in range(len(stage_names)):
        ax.barh(y_pos[i], means[i], left=starts[i], height=0.7, color=colors[i], edgecolor='black')
        
        x_dot = starts[i] + means[i]
        
        ax.errorbar(x_dot, y_pos[i], xerr=stds[i], fmt='ko', capsize=0, zorder=3, 
                    ecolor='black', elinewidth=1.5, markersize=5)
        
    # 4. Formatierung der Achsen
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (us)')
    
    ax.plot([], [], 'ko', label='Std Dev')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
    
    # 5. Titel formatieren (Angepasst an die neuen Ordnernamen)
    arch = folder_path.name.replace("Timing_", "") # Fallback
    solver = "PCG" 
    knots = "?"
    
    match_knots = re.search(r"_K(\d+)", folder_path.name)
    match_solver = re.search(r"_(pcg|qdldl)", folder_path.name, re.IGNORECASE)
    
    if match_knots: 
        knots = match_knots.group(1)
        
    if match_solver:
        solver = match_solver.group(1).upper()
        # Architektur ist alles zwischen "Timing_" und dem Solver-Namen
        arch_match = re.search(rf"Timing_(.+?)_{match_solver.group(1)}", folder_path.name, re.IGNORECASE)
        if arch_match:
            arch = arch_match.group(1)
            
    # Den finalen Titel dynamisch zusammenbauen
    title_parts = [f"{arch}", f"{solver}", f"n={knots}"]
    
    # Toleranz nur bei PCG im Titel anzeigen
    if solver == "PCG":
        title_parts.append(f"Tol={tolerance}")
        
    title = " | ".join(title_parts)
    ax.set_title(f"Component Analysis MPCGPU\n{title}", pad=15)
    
    plt.tight_layout()
    
    # 6. Speichern
    output_dir = folder_path.parent.parent / "timing_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"{folder_path.name}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Plot generiert: {save_path.name}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "timing_results"
    
    if not base_dir.exists():
        print(f"Ordner {base_dir} nicht gefunden. Hast du die Benchmarks schon laufen lassen?")
    else:
        print("===================================")
        print("=== Starte Plotting             ===")
        print("===================================")
        for run_dir in base_dir.iterdir():
            if run_dir.is_dir():
                plot_gantt_chart(run_dir)
        print("\n===================================")
        print("=== Plotting abgeschlossen!     ===")
        print("===================================")
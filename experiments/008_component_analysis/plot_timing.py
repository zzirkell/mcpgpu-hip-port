#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stage_data(folder_path, stage_keyword, tolerance="0.000050"):
    """
    Sucht nach ALLEN Dateien, die die gewählte Toleranz UND das Keyword enthalten,
    liest alle Zeilen aus allen Iterationen ein und kombiniert sie.
    """
    # HIER ist die Magie: Er sucht jetzt z.B. nach *0.000050*ktt_times*.result
    search_pattern = f"*{tolerance}*{stage_keyword}*.result"
    files = list(folder_path.glob(search_pattern))
    
    data = []
    if not files:
        return np.array(data)
        
    for file_path in files:
        with open(file_path, 'r') as file:
            for line in file:
                # Entferne Leerzeichen und eventuelle Kommas
                val = line.strip().replace(',', '')
                if val:
                    try:
                        data.append(float(val))
                    except ValueError:
                        pass
                        
    return np.array(data)

def plot_gantt_chart(folder_path):
    # Diese Namen müssen exakt mit den Dateinamen-Bausteinen übereinstimmen
    stages = ["ktt_times", "shur_times", "linsys_times", "dz_times", "line_search_times"]
    stage_names = ["KKT", "Schur", "Linsys", "DZ", "Line Search"]

    tolerance = "0.000050"
    
    # Farben so nah wie möglich am Original-Paper
    colors = ['#ADD8E6', '#90EE90', '#F08080', '#FFFACD', '#87CEFA'] 
    
    means = []
    stds = []
    
    # 1. Daten laden und Statistiken berechnen
    for stage in stages:
        # Hier übergeben wir die gewünschte Toleranz
        data = load_stage_data(folder_path, stage, tolerance)
        
        if len(data) > 0:
            means.append(np.mean(data))
            stds.append(np.std(data))
        else:
            print(f"[!] Warnung: Keine Daten für '{stage}' mit Tol 5e-5 in {folder_path.name} gefunden.")
            means.append(0.0)
            stds.append(0.0)
            
    # Falls der Ordner für diese Toleranz komplett leer ist, abbrechen
    if sum(means) == 0:
        print(f"[!] Überspringe {folder_path.name} (Keine Timing-Daten für Tol 5e-5 gefunden).")
        return
        
    # 2. Den Plot vorbereiten
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Y-Positionen (Wir drehen sie um, damit KKT ganz oben steht)
    y_pos = np.arange(len(stage_names))[::-1] 
    
    # Startpunkte (X-Achse) für die diagonal gestapelten Balken berechnen
    starts = [0]
    for i in range(len(means)-1):
        starts.append(starts[-1] + means[i])
        
    # Y-Achsen Labels bauen inkl. Mean & Std (z.B. "KKT (37.44 ± 0.75)")
    y_labels = [f"{name} ({m:.2f} ± {s:.2f})" for name, m, s in zip(stage_names, means, stds)]
    
    # 3. Zeichnen der Elemente
    for i in range(len(stage_names)):
        # Den horizontalen Balken zeichnen
        ax.barh(y_pos[i], means[i], left=starts[i], height=0.7, color=colors[i], edgecolor='black')
        
        # Den Punkt (Mean) exakt am rechten Rand des Balkens platzieren und die Std-Dev als Linie zeichnen
        x_dot = starts[i] + means[i]
        
        # Die schwarze horizontale Linie für die Standardabweichung
        ax.errorbar(x_dot, y_pos[i], xerr=stds[i], fmt='ko', capsize=0, zorder=3, 
                    ecolor='black', elinewidth=1.5, markersize=5)
        
    # 4. Formatierung der Achsen
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (us)')
    
    # Legende für den Standardabweichungs-Punkt
    ax.plot([], [], 'ko', label='Std Dev')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
    
    # Titel formatieren
    raw_title = folder_path.name.replace("Timing_", "")
    parts = raw_title.split("_K")
    if len(parts) == 2:
        arch = parts[0]
        sub_parts = parts[1].split("_WS_")
        if len(sub_parts) == 2:
            knots = sub_parts[0]
            ws = sub_parts[1]
            if arch != "cuda":
                title = f"SQP Iteration Timing Breakdown - {arch}, n={knots}, tolerance={tolerance}, WS={ws}"
            else:
                title = f"SQP Iteration Timing Breakdown - {arch}, n={knots}, tolerance={tolerance}"
        else:
            title = f"SQP Iteration Timing Breakdown - {raw_title}"
    else:
        title = f"SQP Iteration Timing Breakdown - {raw_title}"
        
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Als Grafik direkt in den jeweiligen Result-Ordner speichern
    save_path = folder_path / "timing_breakdown.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot generiert: {save_path.name} in {folder_path.name}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "timing_results"
    
    if not base_dir.exists():
        print(f"Ordner {base_dir} nicht gefunden. Hast du die Benchmarks schon laufen lassen?")
    else:
        print("=== Starte Plotting ===")
        for run_dir in base_dir.iterdir():
            if run_dir.is_dir():
                plot_gantt_chart(run_dir)
        print("=== Plotting abgeschlossen! ===")
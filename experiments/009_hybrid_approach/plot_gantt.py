#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def load_stage_data(folder_path, stage_keyword):
    search_pattern = f"*{stage_keyword}*.result"
    files = list(folder_path.glob(search_pattern))
    
    data = []
    if not files: return np.array(data)
        
    for file_path in files:
        with open(file_path, 'r') as file:
            for line in file:
                val = line.strip().replace(',', '')
                if val:
                    try: data.append(float(val))
                    except ValueError: pass
    return np.array(data)

def get_hybrid_stats(folder_path):
    winners = load_stage_data(folder_path, "hybrid_winners")
    if len(winners) > 0:
        qdldl_wins = np.count_nonzero(winners == 1)
        pcg_wins = np.count_nonzero(winners == 2)
        timeouts = np.count_nonzero(winners == 0)

        return f"Hybrid Race Wins:   QDLDL (CPU) = {qdldl_wins}   |   PCG (GPU) = {pcg_wins}   |   Timeouts = {timeouts}"
    return ""

def plot_gantt_chart(folder_path):
    stages = ["ktt_times", "shur_times", "linsys_times", "dz_times", "line_search_times"]
    stage_names = ["KKT", "Schur", "Linsys", "DZ", "Line Search"]
    colors = ['#ADD8E6', '#90EE90', '#F08080', '#FFFACD', '#87CEFA'] 
    
    means, stds = [], []
    print(f"\n--- Analysiere Ordner: {folder_path.name} ---")
    
    stats_text = get_hybrid_stats(folder_path)
    if stats_text:
        print(f"  [>] {stats_text}")
    
    for stage in stages:
        data = load_stage_data(folder_path, stage)
        if len(data) > 0:
            means.append(np.mean(data))
            stds.append(np.std(data))
        else:
            means.append(0.0); stds.append(0.0)
            
    if sum(means) == 0: return
        
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y_pos = np.arange(len(stage_names))[::-1] 
    
    starts = [0]
    for i in range(len(means)-1):
        starts.append(starts[-1] + means[i])
        
    y_labels = [f"{name} ({m:.2f} ± {s:.2f})" for name, m, s in zip(stage_names, means, stds)]
    
    for i in range(len(stage_names)):
        ax.barh(y_pos[i], means[i], left=starts[i], height=0.7, color=colors[i], edgecolor='black')
        ax.errorbar(starts[i] + means[i], y_pos[i], xerr=stds[i], fmt='ko', capsize=0, zorder=3, ecolor='black')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    
    if stats_text:
        ax.set_xlabel(f"Time (us)\n\n{stats_text}", weight="bold")
    else:
        ax.set_xlabel("Time (us)")
    
    match_knots = re.search(r"_K(\d+)", folder_path.name)
    match_solver = re.search(r"_(pcg|qdldl|hybrid)", folder_path.name, re.IGNORECASE)
    
    knots = match_knots.group(1) if match_knots else "?"
    solver = match_solver.group(1).upper() if match_solver else "HYBRID"
    
    ax.set_title(f"Component Analysis | Arch: CUDA | Solver: {solver} | n={knots}", pad=15)
    
    plt.tight_layout()
    output_dir = folder_path.parent.parent / "timing_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"{folder_path.name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "timing_data"
    for run_dir in base_dir.iterdir():
        if run_dir.is_dir():
            plot_gantt_chart(run_dir)
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from pathlib import Path

def parse_folder_name(folder_name):
    pattern = re.compile(r"Run_(\d+)_(.+?)_(pcg|qdldl|hybrid)_K(\d+)_B(\d+)_WS(ON|OFF)", re.IGNORECASE)
    match = pattern.search(folder_name)
    if match:
        budget_us = int(match.group(5))
        freq_hz = int(1000000 / budget_us) if budget_us > 0 else 0
        return {
            "Arch": match.group(2).upper(),
            "Solver": match.group(3).upper(),
            "Knots": int(match.group(4)),
            "Freq": f"{freq_hz}Hz" if freq_hz < 1000 else "1kHz",
            "Budget": budget_us
        }
    return None

def plot_heatmaps():
    base_dir = Path(__file__).resolve().parent / "heatmap_data"
    all_data = []

    if not base_dir.exists():
        print("[!] Ordner heatmap_data existiert nicht.")
        return

    for exp_folder in base_dir.iterdir():
        if not exp_folder.is_dir(): continue
        params = parse_folder_name(exp_folder.name)
        if not params: continue
        
        for csv_file in exp_folder.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 1:
                    metric_value = float(df.iloc[1, 0]) 
                    if metric_value > 0:
                        params["SQP_Iters"] = params["Budget"] / metric_value
                    else:
                        params["SQP_Iters"] = float('nan')
                    all_data.append(params)
            except Exception: pass

    df = pd.DataFrame(all_data)
    if df.empty:
        print("[!] Keine Daten zum Plotten gefunden.")
        return

    image_groups = {
        "CUDA": [
            (("CUDA", "QDLDL"), "QDLDL"),
            (("CUDA", "PCG"), "GBD-PCG")
        ],
        "NV_HIP": [
            (("NV_HIP", "QDLDL"), "QDLDL"),
            (("NV_HIP", "PCG"), "GBD-PCG")
        ],
        "AMD_HIP": [
            (("AMD_HIP", "QDLDL"), "QDLDL"),
            (("AMD_HIP", "PCG"), "GBD-PCG")
        ],
        "HYBRID_CUDA": [
            (("CUDA", "HYBRID"), "HYBRID")
        ]
    }

    freq_order = ["250Hz", "500Hz", "1kHz"]
    knot_order = [32, 64, 128, 256, 512]
    cmap = 'RdYlGn'

    for system_name, configs in image_groups.items():
        active_configs = []
        for (arch, solver), solver_label in configs:
            if not df[(df["Arch"] == arch) & (df["Solver"] == solver)].empty:
                active_configs.append(((arch, solver), solver_label))

        num_plots = len(active_configs)
        if num_plots == 0:
            continue

        print(f"[+] Zeichne {num_plots} Heatmap(s) für {system_name}...")

   
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))
        if num_plots == 1: axes = [axes]
        
        fig.suptitle(f'MPCGPU Performance - {system_name.upper()} (WS_ON)', fontsize=16, weight='bold')

        for i, ((arch, solver), solver_label) in enumerate(active_configs):
            ax = axes[i]
            subset = df[(df["Arch"] == arch) & (df["Solver"] == solver)]
            
            pivot = subset.pivot(index="Freq", columns="Knots", values="SQP_Iters")
            pivot = pivot.reindex(index=freq_order, columns=knot_order)

           
            ax.set_facecolor('#e0e0e0')

            sns.heatmap(pivot.astype(float), ax=ax, cmap=cmap, 
                        norm=colors.LogNorm(vmin=1.0, vmax=25.0), 
                        annot=True, fmt=".1f", 
                        annot_kws={"fontsize": 13, "weight": "bold"},
                        linewidths=1, linecolor='gray',
                        cbar_kws={'label': 'Iterations (Period / Avg Solve Time)'})

           
            for y_idx, (rate_label, row) in enumerate(pivot.iterrows()):
                for x_idx, (knot_label, val) in enumerate(row.items()):
                    if pd.isna(val) or val < 1.0:
                        ax.add_patch(plt.Rectangle((x_idx, y_idx), 1, 1, fill=True, color='#c93c20', alpha=1.0))
                        ax.text(x_idx + 0.5, y_idx + 0.5, 'X', 
                                ha='center', va='center', fontsize=18, color='black', weight='bold')

           
            ax.set_title(f'MPCGPU with {solver_label}', loc='left', fontsize=16, weight='bold', pad=20, color='white',
                         bbox=dict(facecolor='black', edgecolor='black', boxstyle='square,pad=0.5'))
            
           
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Knot Points', fontsize=14, weight='bold', labelpad=10)
            ax.xaxis.tick_top()
            ax.set_xticklabels([str(k) for k in knot_order], fontsize=14, weight='bold')
            
            ax.set_ylabel('Control Rate', fontsize=14, weight='bold', labelpad=10)
            ax.set_yticklabels(freq_order, fontsize=14, weight='bold', rotation=90, va='center')

       
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_file = f"Heatmap_{system_name}_benchmark.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Plot successfully saved to: {out_file}\n")

if __name__ == "__main__":
    plot_heatmaps()
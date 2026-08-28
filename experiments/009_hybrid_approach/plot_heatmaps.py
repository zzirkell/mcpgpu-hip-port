#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
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
            "Freq": f"{freq_hz}Hz"
        }
    return None

def plot_heatmaps():
    base_dir = Path(__file__).resolve().parent / "heatmap_data"
    all_data = []

    if not base_dir.exists():
        print("[!] Folder heatmap_data is missing.")
        return

    # Daten einlesen
    for exp_folder in base_dir.iterdir():
        if not exp_folder.is_dir(): continue
        params = parse_folder_name(exp_folder.name)
        if not params: continue
        
        for csv_file in exp_folder.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 1:
                    metric_value = df.iloc[1, 0] 
                    params["SQP_Iters"] = float(metric_value)
                    all_data.append(params)
            except Exception: pass

    df = pd.DataFrame(all_data)
    if df.empty:
        print("[!] No data found.")
        return

    image_groups = {
        "Heatmaps_CUDA": [
            (("CUDA", "PCG"), "CUDA with PCG"),
            (("CUDA", "QDLDL"), "CUDA with QDLDL")
        ],
        "Heatmaps_NV_HIP": [
            (("NV_HIP", "PCG"), "NVIDIA HIP with PCG"),
            (("NV_HIP", "QDLDL"), "NVIDIA HIP with QDLDL")
        ],
        "Heatmaps_AMD_HIP": [
            (("AMD_HIP", "PCG"), "AMD HIP with PCG"),
            (("AMD_HIP", "QDLDL"), "AMD HIP with QDLDL")
        ],
        "Heatmaps_HYBRID": [
            (("CUDA", "HYBRID"), "CUDA with HYBRID")
        ]
    }

    freq_order = ["250Hz", "500Hz", "1kHz"]
    knot_order = [32, 64, 128, 256, 512]

    for image_name, configs in image_groups.items():
        active_configs = []
        for (arch, solver), title in configs:
            if not df[(df["Arch"] == arch) & (df["Solver"] == solver)].empty:
                active_configs.append(((arch, solver), title))

        num_plots = len(active_configs)
        if num_plots == 0:
            continue 

        print(f"[+] Zeichne {num_plots} Heatmap(s) für {image_name}...")

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3.5 * num_plots))
        if num_plots == 1: axes = [axes]

        for i, ((arch, solver), title) in enumerate(active_configs):
            subset = df[(df["Arch"] == arch) & (df["Solver"] == solver)]
            
            pivot = subset.pivot(index="Freq", columns="Knots", values="SQP_Iters")
            pivot = pivot.reindex(index=freq_order, columns=knot_order)

            sns.heatmap(pivot.astype(float), ax=axes[i], annot=True, fmt=".1f", 
                        cmap="RdYlGn_r", cbar=False, linewidths=1, linecolor='gray',
                        vmin=1.0, vmax=25.0) 
            
            for (j, k), val in np.ndenumerate(pivot.values):
                if pd.isna(val):
                    axes[i].text(k+0.5, j+0.5, 'X', ha='center', va='center', color='black', weight='bold')

            axes[i].set_title(title, weight='bold', fontsize=14, pad=10)
            axes[i].set_xlabel("Knot Points", weight='bold')
            axes[i].set_ylabel("Control Rate", weight='bold')
            axes[i].tick_params(axis='both', which='major', labelsize=11)

        plt.tight_layout()
        out_file = f"{image_name}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Stored : {out_file}")

if __name__ == "__main__":
    plot_heatmaps()
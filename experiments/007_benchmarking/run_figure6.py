import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

# configuration
BASE_DIRS = ["benchmark_archive_ws_on", "benchmark_archive_ws_off"]

OUTPUT_DIR = "figure6_results"
SYSTEMS = ["cuda", "nv_hip", "amd_hip"]

SOLVERS_MAPPING = {
    "QDLDL": "QDLDL", 
    "PCG": "GBD-PCG"
}

KNOT_POINTS = [32, 64, 128, 256, 512]
CONTROL_RATES_HZ = [250, 500, 1000]
PCG_TARGET_TOLERANCE = "0.000007" 

def create_fig6_for_system(system_name, base_dir):
    """Generates the Fig. 6 heatmap for a specific system and workspace configuration."""
    system_dir = os.path.join(base_dir, system_name)
    
    if not os.path.exists(system_dir):
        return

    ws_label = base_dir.replace("benchmark_archive_", "").upper() 
    print(f"--- Creating plot for system: {system_name} ({ws_label}) ---")
    
    fig, axes = plt.subplots(len(SOLVERS_MAPPING), 1, figsize=(10, 12))
    
    fig.suptitle(f'MPCGPU Performance (Fig. 6) - {system_name.upper()} ({ws_label})', fontsize=16, weight='bold')
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True, s=90, l=60)

    for i, (solver_folder, solver_label) in enumerate(SOLVERS_MAPPING.items()):
        heatmap_matrix = pd.DataFrame(index=CONTROL_RATES_HZ, columns=KNOT_POINTS, data=np.nan)
        ax = axes[i]
        
        for rate in CONTROL_RATES_HZ:
            period_ms = 1000.0 / rate 
            
            for knot in KNOT_POINTS:
                folder_path = os.path.join(system_dir, solver_folder, f"knots_{knot}_rate_{rate}")
                
                if solver_folder == "PCG":
                    search_pattern = os.path.join(folder_path, f"*_PCG_{PCG_TARGET_TOLERANCE}_overall_stats.csv")
                else:
                    search_pattern = os.path.join(folder_path, "*_QDLDL_overall_stats.csv")
                
                csv_files = glob.glob(search_pattern)
                
                if not csv_files:
                    heatmap_matrix.loc[rate, knot] = np.nan
                    continue
                
                csv_file = csv_files[0] 
                
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [col.strip() for col in df.columns]
                    
                    iterations = float(df.iloc[1, 0])
                    
                    if iterations > 0:
                        if iterations >= 1.0:
                            heatmap_matrix.loc[rate, knot] = iterations
                        else:
                            heatmap_matrix.loc[rate, knot] = np.nan
                    else:
                         heatmap_matrix.loc[rate, knot] = np.nan
                         
                except Exception as e:
                    print(f"    [!] Error reading {csv_file}: {e}")
                    heatmap_matrix.loc[rate, knot] = np.nan

        heatmap_matrix = heatmap_matrix.sort_index(ascending=True)

        sns.heatmap(heatmap_matrix, 
                    ax=ax, 
                    cmap=cmap, 
                    norm=colors.LogNorm(vmin=1.0, vmax=25.0), 
                    annot=True, fmt=".1f", 
                    annot_kws={"fontsize": 13, "weight": "bold"},
                    linewidths=1, linecolor='gray',
                    cbar_kws={'label': 'Iterations (Period / Avg Solve Time)'})

        for y_idx, (rate_label, row) in enumerate(heatmap_matrix.iterrows()):
            for x_idx, (knot_label, val) in enumerate(row.items()):
                if pd.isna(val):
                    ax.add_patch(plt.Rectangle((x_idx, y_idx), 1, 1, fill=True, color='#c93c20', alpha=1.0))
                    ax.text(x_idx + 0.5, y_idx + 0.5, 'X', 
                            ha='center', va='center', fontsize=18, color='black', weight='bold')

        ax.set_title(f'MPCGPU with {solver_label}', loc='left', fontsize=16, weight='bold', pad=20, color='white',
                     bbox=dict(facecolor='black', edgecolor='black', boxstyle='square,pad=0.5'))
        
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Knot Points', fontsize=14, weight='bold', labelpad=10)
        ax.xaxis.tick_top()
        ax.set_xticklabels([str(k) for k in KNOT_POINTS], fontsize=14, weight='bold')
        
        ax.set_ylabel('Control Rate', fontsize=14, weight='bold', labelpad=10)
        yticklabels = [f"{int(r/1000)}kHz" if r >= 1000 else f"{int(r)}Hz" for r in heatmap_matrix.index]
        ax.set_yticklabels(yticklabels, fontsize=14, weight='bold', rotation=90, va='center')
        
        ax.set_facecolor('#e0e0e0')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = os.path.join(OUTPUT_DIR, f"Fig6_{system_name}_{ws_label}_benchmark.png")
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    print(f"[+] Plot successfully saved to: {output_filename}\n")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for base_dir in BASE_DIRS:
        if not os.path.exists(base_dir):
            print(f"[*] Verzeichnis {base_dir} nicht gefunden. Wird übersprungen.")
            continue
            
        for sys_name in SYSTEMS:
            create_fig6_for_system(sys_name, base_dir)
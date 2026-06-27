import pandas as pd
import numpy as np
import subprocess
import os, shutil
import re

# 1. Config
INPUT_CSV = "filter_new.csv"
OUT_DAT_SPEEDUP = "data_speedup_trends.dat"
OUT_DAT_BREAKDOWN = "data_breakdown_trends.dat"
OUT_GP_SPEEDUP = "plot_speedup_trends.gp"
OUT_GP_BREAKDOWN = "plot_breakdown_trends.gp"
OUT_EPS_SPEEDUP = "avg_speedup_trends.eps"
OUT_EPS_BREAKDOWN = "avg_phase_breakdown_trends.eps"

# --- VISUAL SETTINGS ---
PNG_FONT_SIZE = 32
PNG_KEY_FONT_SIZE = 24
LINE_WIDTH = 4

def extract_id(melody):
    m = re.search(r"[dac](\d+)", str(melody).lower())
    return m.group(1) if m else "00"

def process_data_global_speedup():
    print(f"Reading {INPUT_CSV} for Global Speedup...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        return None

    if "Run_ID" not in df.columns:
        return None

    df["Run_ID"] = df["Run_ID"].astype(str)
    df_runs = df[df["Run_ID"].str.match(r"^\d+$")].copy()
    
    col = "Speedup"
    if col in df_runs.columns:
        df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')
    else:
        df_runs[col] = 0.0

    group_cols = ["Bars"]
    grouped = df_runs.groupby(group_cols)["Speedup"].mean().reset_index()
    return grouped.sort_values("Bars")

def process_data_breakdown_3pt():
    print(f"Reading {INPUT_CSV} for Breakdown...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return None

    df["Run_ID"] = df["Run_ID"].astype(str)
    df_runs = df[df["Run_ID"].str.match(r"^\d+$")].copy()
    
    numeric_cols = ["Total_Fast", "Total_Slow", "Speedup"]
    phase_cols = ["Fast_init", "Slow_init", 
                  "Fast_selection", "Slow_selection", 
                  "Fast_crossover", "Slow_crossover", 
                  "Fast_mutation", "Slow_mutation", 
                  "Fast_evaluation", "Slow_evaluation", 
                  "Fast_survivor", "Slow_survivor"]
    
    for c in numeric_cols + phase_cols:
        if c in df_runs.columns:
            df_runs[c] = pd.to_numeric(df_runs[c], errors='coerce')
        else:
            df_runs[c] = 0.0

    group_cols = ["Style", "Melody", "Bars"]
    grouped = df_runs.groupby(group_cols).mean(numeric_only=True).reset_index()
    
    grouped["ID"] = grouped["Melody"].apply(extract_id)
    grouped["Label"] = grouped["Style"] + grouped["ID"] + " (H = " + grouped["Bars"].astype(str) + ")"
    
    grouped = grouped.sort_values("Speedup")
    n = len(grouped)
    if n == 0:
        return None
        
    min_row = grouped.iloc[0]
    max_row = grouped.iloc[-1]
    mid_idx = n // 2
    mid_row = grouped.iloc[mid_idx]
    
    return pd.DataFrame([min_row, mid_row, max_row])

def write_speedup_dat(df):
    print(f"Writing {OUT_DAT_SPEEDUP}...")
    with open(OUT_DAT_SPEEDUP, "w") as f:
        f.write("# Bars Speedup\n")
        for _, row in df.iterrows():
            f.write(f'{row["Bars"]} {row["Speedup"]:.4f}\n')

def write_breakdown_dat(df):
    print(f"Writing {OUT_DAT_BREAKDOWN}...")
    
    rows = [df.iloc[0], df.iloc[1], df.iloc[2]]
    titles = ["Min Speedup", "Med Speedup", "Max Speedup"]
    
    with open(OUT_DAT_BREAKDOWN, "w") as f:
        f.write("# Label Mode F_Init ... S_Surv\n")
        
        for i, row in enumerate(rows):
            lbl_base = titles[i]
            label_str = f"{lbl_base} ({row['Speedup']:.2f}x)\\n{row['Label']}"
            
            # Slow Row (Mode 0)
            f.write(f'"{label_str}" 0 '
                    f'0 0 0 0 0 0 '
                    f'{row["Slow_init"]:.4f} {row["Slow_selection"]:.4f} {row["Slow_crossover"]:.4f} '
                    f'{row["Slow_mutation"]:.4f} {row["Slow_evaluation"]:.4f} {row["Slow_survivor"]:.4f}\n')
            
            # Fast Row (Mode 1)
            f.write(f'"{label_str}" 1 '
                    f'{row["Fast_init"]:.4f} {row["Fast_selection"]:.4f} {row["Fast_crossover"]:.4f} '
                    f'{row["Fast_mutation"]:.4f} {row["Fast_evaluation"]:.4f} {row["Fast_survivor"]:.4f} '
                    f'0 0 0 0 0 0\n')
            # Gap
            if i < 2:
                 f.write(f'"" -1 0 0 0 0 0 0 0 0 0 0 0 0\n')

def write_speedup_gp():
    print(f"Writing {OUT_GP_SPEEDUP}...")
    script = f"""
# --- 1. EPS OUTPUT ---
set terminal postscript eps enhanced color font 'Helvetica,36' size 10,6
set output '{OUT_EPS_SPEEDUP}'

set title "Average Speedup vs Harmonization Length"
set ylabel "Speedup"
set xlabel "Harmonization Length (H)"
set grid
set yrange [1:*]

# Matched Margins (Tightened)
set lmargin 8
set bmargin 3.0  
set rmargin 2
set tmargin 2

set key top left spacing 1.2 opaque box

plot '{OUT_DAT_SPEEDUP}' using 1:2 with linespoints lw {LINE_WIDTH} pt 7 ps 2 lc rgb '#CC3311' title 'Speedup'

# --- 2. PNG OUTPUT ---
set terminal pngcairo size 1800,1200 enhanced font 'Sans,{PNG_FONT_SIZE}'
set output '{OUT_EPS_SPEEDUP.replace('.eps', '.png')}'

# Matched Margins for PNG (Tightened)
set lmargin 9
set bmargin 4.0
set rmargin 3
set tmargin 2

set key font ",{PNG_KEY_FONT_SIZE}" spacing 1.3

replot
"""
    with open(OUT_GP_SPEEDUP, "w") as f:
        f.write(script)

def write_breakdown_gp_final(df):
    base_script = write_breakdown_gp_script_content(df)
    with open(OUT_GP_BREAKDOWN, "w") as f:
        f.write(base_script)

def write_breakdown_gp_script_content(df):
    labels = []
    titles_list = ["Min Speedup", "Med Speedup", "Max Speedup"]
    for i in range(3):
        row = df.iloc[i]
        lbl = f"{titles_list[i]} ({row['Speedup']:.2f}x)\\n{row['Label']}"
        labels.append(lbl)
    
    colors = ["#BBBBBB", "#EE7733", "#0077BB", "#33BBEE", "#CC3311", "#009988"]
    titles = ["Init", "Selection", "Crossover", "Mutation", "Evaluation", "Survivor"]
    
    cmds = []
    
    # 1. Slow Columns (Full) - Renamed "Slow"
    for i in range(6):
        col = 9 + i
        t = f"Slow {titles[i]}"
        c = colors[i]
        cmds.append(f"'{OUT_DAT_BREAKDOWN}' using {col} title '{t}' lc rgb '{c}' fs pattern 4 border -1")
        
    # 2. Fast Columns (Partial) - Renamed "Fast"
    for i in range(6):
        col = 3 + i
        t = f"Fast {titles[i]}"
        c = colors[i]
        cmds.append(f"'' using {col} title '{t}' lc rgb '{c}' fs solid 1.0 border -1")
        
    plot_cmd_str = ", \\\n     ".join(cmds)
    
    return f"""
# --- 1. EPS OUTPUT ---
set terminal postscript eps enhanced color font 'Helvetica,36' size 10,6
set output '{OUT_EPS_BREAKDOWN}'

set title "Runtime Breakdown"
set ylabel "Total Time (s)"

# Margins: bmargin 4.0 is just enough for 2-line labels in EPS
set lmargin 8
set bmargin 3.0 
set rmargin 2
set tmargin 2

set style data histogram
set style histogram rowstacked
set boxwidth 0.8

set xtics ("{labels[0]}" 0.5, "{labels[1]}" 3.5, "{labels[2]}" 6.5) scale 0

set grid y
set key top left inside reverse Left columns 2 width -2 samplen 2 opaque box spacing 1.1

plot {plot_cmd_str}

# --- 2. PNG OUTPUT ---
set terminal pngcairo size 1800,1200 enhanced font 'Sans,{PNG_FONT_SIZE}'
set output '{OUT_EPS_BREAKDOWN.replace('.eps', '.png')}'

# Margins: bmargin 5.0 is safe for 2-line labels in PNG (Size 32)
set lmargin 9
set bmargin 4.0 
set rmargin 3
set tmargin 2

set key font ",{PNG_KEY_FONT_SIZE}" spacing 1.2

replot
"""

def run_gnuplot():
    gnuplot_cmd = shutil.which("gnuplot")
    if not gnuplot_cmd:
        print("Warning: gnuplot not found.")
        return
    for gp in [OUT_GP_SPEEDUP, OUT_GP_BREAKDOWN]:
        subprocess.run([gnuplot_cmd, gp], check=True)

if __name__ == "__main__":
    df_speedup = process_data_global_speedup()
    if df_speedup is not None:
        write_speedup_dat(df_speedup)
        write_speedup_gp()
    
    df_breakdown = process_data_breakdown_3pt()
    if df_breakdown is not None:
        write_breakdown_dat(df_breakdown)
        write_breakdown_gp_final(df_breakdown)
        
    run_gnuplot()
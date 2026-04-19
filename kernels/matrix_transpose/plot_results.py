#!/usr/bin/env python3
"""
plot_results.py
===============
Generates 3 focused comparison plots for the parallel reduction benchmark.

Usage (with real data):
    python3 plot_results.py \
        --nvidia-results results_nvidia.csv \
        --amd-results    results_amd.csv \
        --nvidia-sweep   sweep_nvidia.csv \
        --amd-sweep      sweep_amd.csv

Usage (demo/test without running on HPC):
    python3 plot_results.py --demo

Output:
    fig1_bandwidth.png   - Achieved BW + % of peak BW (the key cross-platform metric)
    fig2_sweep.png       - Bandwidth vs array size (shows roofline saturation)
    fig3_speedup_roof.png - Speedup bars + Roofline model
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ──────────────────────────────────────────────────────────────
NV_COLORS  = ["#4CAF50", "#2196F3", "#FF5722"]   # green, blue, orange-red
AMD_COLORS = ["#9C27B0", "#FF9800", "#00BCD4"]   # purple, orange, cyan
NV_LINE    = "#76b900"   # NVIDIA green
AMD_LINE   = "#ed1c24"   # AMD red

PEAK_BW  = {"NVIDIA A100": 1555.0, "AMD MI50": 1024.0}   # GB/s
K_LABELS = ["K1\nBaseline", "K2\nShared Mem", "K3\nShuffle"]

# ── Demo data ──────────────────────────────────────────────────────────
def demo_data():
    nv = pd.DataFrame({
        "kernel_id": [1, 2, 3], "time_ms": [4.82, 1.21, 0.98],
        "bandwidth_GBs": [138.5, 552.1, 681.7], "pct_peak_bw": [6.8, 27.1, 33.4],
        "speedup_vs_baseline": [1.0, 3.98, 4.92], "correct": [1, 1, 1],
    })
    am = pd.DataFrame({
        "kernel_id": [1, 2, 3], "time_ms": [9.10, 2.73, 2.10],
        "bandwidth_GBs": [73.4, 244.7, 318.1], "pct_peak_bw": [7.2, 23.9, 31.1],
        "speedup_vs_baseline": [1.0, 3.33, 4.33], "correct": [1, 1, 1],
    })
    sizes = [2**16, 2**18, 2**20, 2**22, 2**24, 2**26]
    bw_n = {1:[12,25,60,120,138,140], 2:[40,90,200,420,552,570], 3:[50,110,260,530,682,700]}
    bw_a = {1:[8,18,42,68,73,75],    2:[25,55,130,210,245,250], 3:[30,68,160,275,318,325]}
    rn, ra = [], []
    for i, sz in enumerate(sizes):
        for k in [1,2,3]:
            rn.append({"array_size":sz,"kernel_id":k,"bandwidth_GBs":bw_n[k][i],
                       "time_ms":sz*4/(bw_n[k][i]*1e9)*1000})
            ra.append({"array_size":sz,"kernel_id":k,"bandwidth_GBs":bw_a[k][i],
                       "time_ms":sz*4/(bw_a[k][i]*1e9)*1000})
    return nv, am, pd.DataFrame(rn), pd.DataFrame(ra)

def savefig(fig, name, out_dir):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
    plt.close(fig)

# ── Figure 1: Bandwidth + % Peak BW (dual y-axis grouped bars) ─────────
def fig1_bandwidth(nv, am, out_dir):
    """
    Left panel : Achieved bandwidth (GB/s) — raw speed
    Right panel: % of peak bandwidth     — fair cross-platform comparator
    
    Why % peak is the right metric:
    A100 peak = 1555 GB/s, MI50 peak = 1024 GB/s.
      Raw GB/s will always favour A100; % peak shows how well each kernel
      uses the available hardware.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(3); w = 0.35

    for ax, col, title, ylabel, ref in [
        (axes[0], "bandwidth_GBs", "Achieved Memory Bandwidth", "GB/s",
         [PEAK_BW["NVIDIA A100"], PEAK_BW["AMD MI50"]]),
        (axes[1], "pct_peak_bw",   "% of Peak Bandwidth", "% of peak",
         [100, 100]),
    ]:
        b1 = ax.bar(x - w/2, nv[col], w, color=NV_COLORS, alpha=0.88,
                    edgecolor="black", linewidth=0.5, label="NVIDIA A100")
        b2 = ax.bar(x + w/2, am[col], w, color=AMD_COLORS, alpha=0.88,
                    edgecolor="black", linewidth=0.5, hatch="//", label="AMD MI50")

        if col == "bandwidth_GBs":
            ax.axhline(PEAK_BW["NVIDIA A100"], color=NV_LINE, ls="--", lw=1.5,
                       label=f"NVIDIA peak {PEAK_BW['NVIDIA A100']:.0f} GB/s")
            ax.axhline(PEAK_BW["AMD MI50"],    color=AMD_LINE, ls=":",  lw=1.5,
                       label=f"AMD peak {PEAK_BW['AMD MI50']:.0f} GB/s")

        for bar in list(b1) + list(b2):
            v = bar.get_height()
            label_str = f"{v:.0f}" if col == "bandwidth_GBs" else f"{v:.1f}%"
            ax.text(bar.get_x() + bar.get_width()/2, v + (v*0.02),
                    label_str, ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x); ax.set_xticklabels(K_LABELS, fontsize=8.5)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.35)

        nv_p = mpatches.Patch(color=NV_COLORS[1], label="NVIDIA A100")
        am_p = mpatches.Patch(facecolor=AMD_COLORS[1], hatch="//", label="AMD MI50")
        handles = [nv_p, am_p]
        if col == "bandwidth_GBs":
            handles += [
                plt.Line2D([0],[0], color=NV_LINE,  ls="--", label=f"NVIDIA peak {PEAK_BW['NVIDIA A100']:.0f} GB/s"),
                plt.Line2D([0],[0], color=AMD_LINE, ls=":",  label=f"AMD peak {PEAK_BW['AMD MI50']:.0f} GB/s"),
            ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_ylim(0, max(nv[col].max(), am[col].max()) * 1.30)

    fig.suptitle("Memory Bandwidth — NVIDIA A100 vs AMD MI50",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    savefig(fig, "fig1_bandwidth.png", out_dir)


# ── Figure 2: Bandwidth vs array size (roofline saturation) ────────────
def fig2_sweep(sweep_nv, sweep_am, out_dir):
    """
    Shows how bandwidth scales with problem size.
    Both platforms plateau at large sizes (memory-bound region of roofline).
    K3 reaches the plateau soonest, showing the shuffle optimisation
    amortises launch overhead faster.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    ls_styles = ["-o", "-s", "-^"]

    for ax, sweep, title, colors, peak in [
        (axes[0], sweep_nv, "NVIDIA A100", NV_COLORS, PEAK_BW["NVIDIA A100"]),
        (axes[1], sweep_am, "AMD MI50",    AMD_COLORS, PEAK_BW["AMD MI50"]),
    ]:
        for kid, color, ls in zip([1, 2, 3], colors, ls_styles):
            sub = sweep[sweep["kernel_id"] == kid].sort_values("array_size")
            x_mb = sub["array_size"] * 4 / (1024**2)
            ax.plot(x_mb, sub["bandwidth_GBs"], ls, color=color,
                    linewidth=2, markersize=6,
                    label=f"K{kid}: {['Baseline','Shared Mem','Shuffle'][kid-1]}")

        ax.axhline(peak, color="black", ls="--", lw=1.2, alpha=0.55,
                   label=f"Peak BW {peak:.0f} GB/s")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Array Size (MB)")
        ax.set_ylabel("Achieved Bandwidth (GB/s)")
        ax.set_title(f"{title}", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Bandwidth vs Problem Size — Roofline Saturation Behaviour\n"
                 "(Both platforms are memory-bound; K3 reaches peak soonest)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    savefig(fig, "fig2_sweep.png", out_dir)


# ── Figure 3: Speedup bars (left) + Roofline model (right) ─────────────
def fig3_speedup_roofline(nv, am, out_dir):
    """
    Left : Speedup of K2 and K3 over K1 within each platform.
           Shows the optimization payoff is similar (~4-5×) on both.
    Right: Roofline model — reduction has low arithmetic intensity (~0.25 FLOP/B)
           so it is firmly memory-bound on both GPUs.
           Points show K3 is closest to the bandwidth ceiling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Speedup bar chart ——————————————————————————————————————————
    ax = axes[0]
    x = np.array([0, 1])   # K2, K3 only (K1 = 1× by definition)
    w = 0.3
    nv_sp = [nv.loc[nv["kernel_id"]==k,"speedup_vs_baseline"].values[0] for k in [2,3]]
    am_sp = [am.loc[am["kernel_id"]==k,"speedup_vs_baseline"].values[0] for k in [2,3]]

    b1 = ax.bar(x - w/2, nv_sp, w, color=[NV_COLORS[1], NV_COLORS[2]],
                alpha=0.88, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, am_sp, w, color=[AMD_COLORS[1], AMD_COLORS[2]],
                alpha=0.88, edgecolor="black", linewidth=0.5, hatch="//")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{bar.get_height():.2f}×", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(["K2", "K3"])
    ax.set_ylabel("Speedup vs K1 Baseline"); ax.set_ylim(0, max(max(nv_sp), max(am_sp)) * 1.25)
    ax.set_title("Speedup over Baseline\n(both platforms, K1=1×)", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    nv_p = mpatches.Patch(color=NV_COLORS[1], label="NVIDIA A100")
    am_p = mpatches.Patch(facecolor=AMD_COLORS[1], hatch="//", label="AMD MI50")
    ax.legend(handles=[nv_p, am_p], fontsize=9)

    # — Roofline model ————————————————————————————————————————————
    ax = axes[1]
    peak_flops = {"NVIDIA A100": 19.5e12, "AMD MI50": 13.4e12}   # FP32 TFLOPS

    AI = np.logspace(-3, 2, 400)
    for plat, color, ls in [("NVIDIA A100", NV_LINE, "-"), ("AMD MI50", AMD_LINE, "--")]:
        roof = np.minimum(peak_flops[plat], PEAK_BW[plat]*1e9 * AI)
        ax.loglog(AI, roof/1e12, color=color, lw=2.5, ls=ls, label=f"{plat} roofline")

    ai_reduction = 0.25   # 1 add / 4 bytes = 0.25 FLOP/B
    ax.axvline(ai_reduction, color="gray", ls=":", lw=1.4,
               label=f"Reduction AI ≈ {ai_reduction} FLOP/B")

    n_elem = 16_777_216
    for df, colors, plat in [(nv, NV_COLORS, "NVIDIA A100"), (am, AMD_COLORS, "AMD MI50")]:
        for row, color, marker in zip(df.itertuples(), colors, ["o","s","^"]):
            gflops = n_elem / (row.time_ms * 1e-3) / 1e12
            ax.plot(ai_reduction, gflops, marker, color=color, markersize=11,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5,
                    label=f"{plat} K{row.kernel_id} ({gflops:.2f} TFLOPS)")

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Roofline Model\n(Reduction is memory-bound — low AI)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(1e-3, 100)

    fig.suptitle("Optimization Impact & Hardware Roofline — NVIDIA A100 vs AMD MI50",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    savefig(fig, "fig3_speedup_roofline.png", out_dir)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia-results", default="./nvidia/results_nvidia.csv")
    parser.add_argument("--amd-results",    default="./amd/results_amd.csv")
    parser.add_argument("--nvidia-sweep",   default="./nvidia/sweep_nvidia.csv")
    parser.add_argument("--amd-sweep",      default="./amd/sweep_amd.csv")
    parser.add_argument("--out-dir",        default="./plots")
    parser.add_argument("--demo",           action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.demo:
        print("[demo mode] using synthetic data")
        nv, am, sweep_nv, sweep_am = demo_data()
    else:
        nv       = pd.read_csv(args.nvidia_results)
        am       = pd.read_csv(args.amd_results)
        sweep_nv = pd.read_csv(args.nvidia_sweep)
        sweep_am = pd.read_csv(args.amd_sweep)

    print(f"Generating plots → {args.out_dir}/")
    fig1_bandwidth(nv, am, args.out_dir)
    fig2_sweep(sweep_nv, sweep_am, args.out_dir)
    fig3_speedup_roofline(nv, am, args.out_dir)
    print("Done — 3 figures saved.")

if __name__ == "__main__":
    main()

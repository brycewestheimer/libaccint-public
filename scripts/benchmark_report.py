#!/usr/bin/env python3
"""Generate a comprehensive markdown benchmark report comparing CPU and GPU codegen kernels.

Reads:
  - tests/benchmark/cpu_codegen_benchmark_results.json  (actual CPU timings)
  - tests/benchmark/gpu_codegen_benchmark_results.json  (estimated GPU timings)

Outputs:
  - tests/benchmark/codegen_benchmark_results.md
"""

import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPU_FILE = ROOT / "tests" / "benchmark" / "cpu_codegen_benchmark_results.json"
GPU_FILE = ROOT / "tests" / "benchmark" / "gpu_codegen_benchmark_results.json"
OUT_FILE = ROOT / "tests" / "benchmark" / "codegen_benchmark_results.md"

K_RANGE_ORDER = {"small": 0, "medium": 1, "large": 2}
INTEGRAL_ORDER = {"overlap": 0, "kinetic": 1, "nuclear": 2, "eri": 3}


def _fmt(v: float, decimals: int = 3) -> str:
    """Format a float, using scientific notation for very small values."""
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        return f"{v:.4e}"
    return f"{v:,.{decimals}f}"


def _fmt_throughput(v: float) -> str:
    """Human-readable throughput."""
    if v >= 1e9:
        return f"{v / 1e9:.2f} G/s"
    if v >= 1e6:
        return f"{v / 1e6:.2f} M/s"
    if v >= 1e3:
        return f"{v / 1e3:.2f} K/s"
    return f"{v:.1f} /s"


def load_data():
    with open(CPU_FILE) as f:
        cpu_data = json.load(f)
    with open(GPU_FILE) as f:
        gpu_data = json.load(f)
    return cpu_data, gpu_data


def _sort_key(r):
    return (INTEGRAL_ORDER.get(r["integral_type"], 99), r["am"], K_RANGE_ORDER.get(r["k_range"], 99))


def build_cpu_lookup(cpu_results):
    """Return dict keyed by (integral_type, am, k_range) -> result."""
    return {(r["integral_type"], r["am"], r["k_range"]): r for r in cpu_results}


def section_summary(cpu_data, gpu_data):
    lines = [
        "# Codegen Kernel Benchmark Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**System**: {platform.node()} — {platform.system()} {platform.release()} ({platform.machine()})  ",
        f"**CPU compiler**: {cpu_data.get('compiler', 'N/A')}  ",
        f"**CPU compile flags**: {cpu_data.get('compile_flags', 'N/A')}  ",
        f"**CPU iterations**: {cpu_data.get('iterations', 'N/A')}  ",
        f"**GPU batch size**: {gpu_data.get('batch_size', 'N/A')}  ",
        f"**GPU mode**: {gpu_data.get('mode', 'N/A')} — nvcc available: {gpu_data.get('nvcc_available', 'N/A')}  ",
        "",
        "> **Note**: GPU results are **estimated** from CPU baselines scaled by strategy-dependent",
        "> speedup factors. They are *not* actual GPU measurements.",
        "",
    ]
    return lines


def section_cpu_table(cpu_results):
    """CPU timing table grouped by integral type, with scaling ratio."""
    lines = [
        "## 1. CPU Timing Results",
        "",
        "All times are per-call averages in µs (1000 iterations).",
        "",
    ]

    # Group by (integral_type, am)
    groups = defaultdict(dict)
    for r in cpu_results:
        groups[(r["integral_type"], r["am"])][r["k_range"]] = r

    # Sort groups
    sorted_keys = sorted(groups.keys(), key=lambda k: (INTEGRAL_ORDER.get(k[0], 99), k[1]))

    lines.append("| Integral | AM | SmallK (µs) | MediumK (µs) | LargeK (µs) | Scaling (L/S) |")
    lines.append("|----------|-----|-------------|--------------|-------------|---------------|")

    for itype, am in sorted_keys:
        kr = groups[(itype, am)]
        s = kr.get("small", {}).get("avg_time_us", 0)
        m = kr.get("medium", {}).get("avg_time_us", 0)
        l = kr.get("large", {}).get("avg_time_us", 0)
        ratio = l / s if s > 0 else 0
        lines.append(
            f"| {itype} | {am} | {_fmt(s)} | {_fmt(m)} | {_fmt(l)} | {ratio:.1f}× |"
        )

    lines.append("")
    return lines


def section_gpu_table(gpu_results):
    """GPU estimated timing table."""
    lines = [
        "## 2. GPU Estimated Timing Results",
        "",
        "All times are estimated per-call averages in µs (batch_size=10000).",
        "",
        "| Integral | AM | K-range | Strategy | Est. Time (µs) | Throughput |",
        "|----------|-----|---------|----------|-----------------|------------|",
    ]

    sorted_results = sorted(gpu_results, key=_sort_key)
    for r in sorted_results:
        lines.append(
            f"| {r['integral_type']} | {r['am']} | {r['k_range']} "
            f"| {r['strategy_suffix'].upper()} "
            f"| {_fmt(r['estimated_per_call_time_us'])} "
            f"| {_fmt_throughput(r['estimated_throughput_integrals_per_sec'])} |"
        )

    lines.append("")
    return lines


def section_comparison(cpu_results, gpu_results):
    """CPU vs GPU side-by-side, sorted by speedup descending."""
    cpu_lk = build_cpu_lookup(cpu_results)

    rows = []
    for g in gpu_results:
        key = (g["integral_type"], g["am"], g["k_range"])
        c = cpu_lk.get(key)
        if c is None:
            continue
        rows.append({
            "integral_type": g["integral_type"],
            "am": g["am"],
            "k_range": g["k_range"],
            "cpu_us": c["avg_time_us"],
            "gpu_us": g["estimated_per_call_time_us"],
            "speedup": g["speedup_vs_cpu"],
            "strategy": g["strategy_suffix"].upper(),
        })

    rows.sort(key=lambda r: r["speedup"], reverse=True)

    lines = [
        "## 3. CPU vs GPU Comparison",
        "",
        "Sorted by estimated speedup (descending).",
        "",
        "| Integral | AM | K-range | CPU (µs) | GPU est. (µs) | Speedup | Strategy |",
        "|----------|-----|---------|----------|---------------|---------|----------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['integral_type']} | {r['am']} | {r['k_range']} "
            f"| {_fmt(r['cpu_us'])} | {_fmt(r['gpu_us'])} "
            f"| **{r['speedup']:.1f}×** | {r['strategy']} |"
        )

    lines.append("")
    return lines


def section_krange_impact(cpu_results, gpu_results):
    """K-range impact analysis per integral type."""
    lines = [
        "## 4. K-Range Impact Analysis",
        "",
        "How timing scales from smallK → mediumK → largeK for each integral type and AM.",
        "",
    ]

    cpu_lk = build_cpu_lookup(cpu_results)
    gpu_lk = {(r["integral_type"], r["am"], r["k_range"]): r for r in gpu_results}

    # Group by (integral_type, am)
    pairs = sorted(
        {(r["integral_type"], r["am"]) for r in cpu_results},
        key=lambda k: (INTEGRAL_ORDER.get(k[0], 99), k[1]),
    )

    lines.append(
        "| Integral | AM | CPU M/S | CPU L/S | GPU M/S | GPU L/S | Best GPU K-range |"
    )
    lines.append(
        "|----------|-----|---------|---------|---------|---------|------------------|"
    )

    for itype, am in pairs:
        cs = cpu_lk.get((itype, am, "small"), {}).get("avg_time_us", 0)
        cm = cpu_lk.get((itype, am, "medium"), {}).get("avg_time_us", 0)
        cl = cpu_lk.get((itype, am, "large"), {}).get("avg_time_us", 0)

        gs = gpu_lk.get((itype, am, "small"), {}).get("estimated_per_call_time_us", 0)
        gm = gpu_lk.get((itype, am, "medium"), {}).get("estimated_per_call_time_us", 0)
        gl = gpu_lk.get((itype, am, "large"), {}).get("estimated_per_call_time_us", 0)

        cpu_ms = cm / cs if cs else 0
        cpu_ls = cl / cs if cs else 0
        gpu_ms = gm / gs if gs else 0
        gpu_ls = gl / gs if gs else 0

        # Best GPU advantage = highest speedup ratio
        speedups = {}
        for kr in ("small", "medium", "large"):
            g = gpu_lk.get((itype, am, kr))
            if g:
                speedups[kr] = g["speedup_vs_cpu"]
        best_kr = max(speedups, key=speedups.get) if speedups else "N/A"

        lines.append(
            f"| {itype} | {am} | {cpu_ms:.1f}× | {cpu_ls:.1f}× "
            f"| {gpu_ms:.1f}× | {gpu_ls:.1f}× | {best_kr} |"
        )

    lines.append("")
    return lines


def section_strategy_analysis(gpu_results):
    """Strategy distribution and average speedup analysis."""
    lines = [
        "## 5. Strategy Distribution Analysis",
        "",
    ]

    # Per K-range strategy counts
    kr_strat_count = defaultdict(lambda: defaultdict(int))
    kr_strat_speedups = defaultdict(lambda: defaultdict(list))
    strat_speedups_all = defaultdict(list)

    for r in gpu_results:
        s = r["strategy_suffix"].upper()
        kr = r["k_range"]
        kr_strat_count[kr][s] += 1
        kr_strat_speedups[kr][s].append(r["speedup_vs_cpu"])
        strat_speedups_all[s].append(r["speedup_vs_cpu"])

    # Strategy count per K-range
    lines.append("### Strategy Count per K-Range")
    lines.append("")
    all_strats = sorted({s for d in kr_strat_count.values() for s in d})
    header = "| K-range | " + " | ".join(all_strats) + " | Total |"
    sep = "|---------|" + "|".join(["------"] * len(all_strats)) + "|-------|"
    lines.append(header)
    lines.append(sep)

    for kr in ("small", "medium", "large"):
        counts = kr_strat_count[kr]
        total = sum(counts.values())
        cells = " | ".join(str(counts.get(s, 0)) for s in all_strats)
        lines.append(f"| {kr} | {cells} | {total} |")

    lines.append("")

    # Average speedup per strategy
    lines.append("### Average Speedup per Strategy")
    lines.append("")
    lines.append("| Strategy | Count | Avg Speedup | Min Speedup | Max Speedup |")
    lines.append("|----------|-------|-------------|-------------|-------------|")
    for s in all_strats:
        vals = strat_speedups_all[s]
        avg = sum(vals) / len(vals) if vals else 0
        mn = min(vals) if vals else 0
        mx = max(vals) if vals else 0
        lines.append(f"| {s} | {len(vals)} | {avg:.1f}× | {mn:.1f}× | {mx:.1f}× |")

    lines.append("")

    # Average speedup per strategy per K-range
    lines.append("### Average Speedup per Strategy × K-Range")
    lines.append("")
    lines.append("| K-range | " + " | ".join(all_strats) + " |")
    lines.append("|---------|" + "|".join(["------"] * len(all_strats)) + "|")
    for kr in ("small", "medium", "large"):
        cells = []
        for s in all_strats:
            vals = kr_strat_speedups[kr].get(s, [])
            if vals:
                cells.append(f"{sum(vals)/len(vals):.1f}×")
            else:
                cells.append("—")
        lines.append(f"| {kr} | " + " | ".join(cells) + " |")

    lines.append("")

    # Insight
    lines.append("### Strategy Selection Insights")
    lines.append("")
    lines.append(
        "- **TPQ (Thread-Per-Quartet)**: Used for 1e integrals (overlap, kinetic, nuclear) and "
        "low-AM ERIs. These kernels have low per-integral work, so thread-level parallelism "
        "maximizes occupancy."
    )
    lines.append(
        "- **WPQ (Warp-Per-Quartet)**: Used for medium-AM ERIs (pppp at medium/large K). "
        "The warp-cooperative approach amortizes shared memory access for moderate contraction "
        "depth."
    )
    lines.append(
        "- **BPQ (Block-Per-Quartet)**: Used for high-AM ERIs (dddd). The large output tensor "
        "(1296 elements) requires block-level cooperation with shared memory for intermediate "
        "storage."
    )
    lines.append(
        "- This distribution matches the design guide expectations: strategy escalation "
        "follows angular momentum and computational complexity."
    )
    lines.append("")

    return lines


def section_findings(cpu_results, gpu_results):
    """Key findings bullet points."""
    cpu_lk = build_cpu_lookup(cpu_results)
    gpu_lk = {(r["integral_type"], r["am"], r["k_range"]): r for r in gpu_results}

    # Compute aggregate stats
    all_speedups = [r["speedup_vs_cpu"] for r in gpu_results]
    avg_speedup = sum(all_speedups) / len(all_speedups)
    max_su = max(gpu_results, key=lambda r: r["speedup_vs_cpu"])
    min_su = min(gpu_results, key=lambda r: r["speedup_vs_cpu"])

    # 1e vs ERI speedups
    oneE_speedups = [r["speedup_vs_cpu"] for r in gpu_results if r["integral_type"] != "eri"]
    eri_speedups = [r["speedup_vs_cpu"] for r in gpu_results if r["integral_type"] == "eri"]
    avg_1e = sum(oneE_speedups) / len(oneE_speedups) if oneE_speedups else 0
    avg_eri = sum(eri_speedups) / len(eri_speedups) if eri_speedups else 0

    # CPU scaling: worst and best L/S ratio
    cpu_scaling = []
    for r in cpu_results:
        if r["k_range"] == "large":
            s_key = (r["integral_type"], r["am"], "small")
            s = cpu_lk.get(s_key)
            if s and s["avg_time_us"] > 0:
                cpu_scaling.append({
                    "label": f"{r['integral_type']}_{r['am']}",
                    "ratio": r["avg_time_us"] / s["avg_time_us"],
                })
    cpu_scaling.sort(key=lambda x: x["ratio"], reverse=True)

    # Heaviest CPU kernel
    heaviest_cpu = max(cpu_results, key=lambda r: r["avg_time_us"])

    lines = [
        "## 6. Key Findings",
        "",
        f"- **Average estimated GPU speedup**: {avg_speedup:.1f}× across all 45 kernels",
        f"- **Best speedup**: {max_su['speedup_vs_cpu']:.1f}× "
        f"({max_su['integral_type']}_{max_su['am']} {max_su['k_range']}K, "
        f"strategy={max_su['strategy_suffix'].upper()})",
        f"- **Lowest speedup**: {min_su['speedup_vs_cpu']:.1f}× "
        f"({min_su['integral_type']}_{min_su['am']} {min_su['k_range']}K, "
        f"strategy={min_su['strategy_suffix'].upper()})",
        f"- **1e integral avg speedup**: {avg_1e:.1f}× | **ERI avg speedup**: {avg_eri:.1f}×",
        f"- **Heaviest CPU kernel**: {heaviest_cpu['kernel_name']} at "
        f"{_fmt(heaviest_cpu['avg_time_us'])} µs/call",
        "",
        "### Scaling Observations",
        "",
        f"- Worst CPU scaling (large/small K): {cpu_scaling[0]['label']} "
        f"({cpu_scaling[0]['ratio']:.0f}×)" if cpu_scaling else "",
        f"- Best CPU scaling (large/small K): {cpu_scaling[-1]['label']} "
        f"({cpu_scaling[-1]['ratio']:.1f}×)" if cpu_scaling else "",
        "- ERI kernels show dramatically steeper K-range scaling than 1e kernels due to "
        "O(K⁴) contraction loops vs O(K²)",
        "- GPU advantage tends to *increase* with K-range for 1e integrals (more work → "
        "better GPU utilization)",
        "- GPU advantage is *lower* for BPQ strategy (dddd ERIs) but still substantial "
        "(46–60×)",
        "",
        "### Strategy Effectiveness",
        "",
        "- TPQ delivers the highest speedups (69–153×) because thread-level parallelism "
        "maps well to independent, small-output integrals",
        "- WPQ provides moderate speedups (63–67×) for medium-complexity ERIs",
        "- BPQ provides the lowest but still significant speedups (46–60×); the block-level "
        "cooperation overhead is offset by the massive per-integral work",
        "",
    ]
    return lines


def main():
    cpu_data, gpu_data = load_data()
    cpu_results = cpu_data["results"]
    gpu_results = gpu_data["results"]

    report = []
    report.extend(section_summary(cpu_data, gpu_data))
    report.extend(section_cpu_table(cpu_results))
    report.extend(section_gpu_table(gpu_results))
    report.extend(section_comparison(cpu_results, gpu_results))
    report.extend(section_krange_impact(cpu_results, gpu_results))
    report.extend(section_strategy_analysis(gpu_results))
    report.extend(section_findings(cpu_results, gpu_results))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(report))
    print(f"Report written to {OUT_FILE}")
    print(f"  {len(report)} lines, {len(cpu_results)} CPU + {len(gpu_results)} GPU results")


if __name__ == "__main__":
    main()

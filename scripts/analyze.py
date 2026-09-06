import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "benchmarks"
REGISTRY_PATH = BENCH_ROOT / "experiments.jsonl"
REPORT_DIR = BENCH_ROOT / "report"

GAMMA = 5
LABEL = {"on": "DSpark (speculative)", "off": "Baseline (non-speculative)"}
COLOR = {"on": "#1d4ed8", "off": "#9ca3af"}
METRIC_NAMES = {
    "ttft": "Time to First Token",
    "itl": "Inter Token Latency",
    "tput": "Output Token Throughput",
    "rps": "Request Throughput",
}
UNITS = {"ttft": "ms", "itl": "ms", "tput": "tok/s", "rps": "req/s"}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })


def load_profile_metrics(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for conc_dir in sorted(run_dir.glob("c*")):
        if not conc_dir.is_dir():
            continue
        try:
            conc = int(conc_dir.name[1:])
        except ValueError:
            continue
        profile = conc_dir / "profile_export_aiperf.json"
        if not profile.exists():
            continue
        try:
            data = json.loads(profile.read_text())
        except json.JSONDecodeError:
            continue
        metrics: dict = {}
        for m in data.get("benchmark_results", {}).get("llm_metrics", []):
            name, pct, value = m.get("name"), m.get("percentile"), m.get("value")
            if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            for key, full in METRIC_NAMES.items():
                if name and full in name and "Per User" not in name:
                    scale = 1000.0 if m.get("unit") in ("s", "sec", "seconds") else 1.0
                    metrics.setdefault(key, {})[pct or "mean"] = value * scale
                    break
        out[conc] = metrics
    return out


def registry() -> None:
    runs = []
    for run_dir in sorted(BENCH_ROOT.iterdir()):
        config_path = run_dir / "run-config.json"
        summary_path = run_dir / "sweep-summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        config = json.loads(config_path.read_text())
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        profile = load_profile_metrics(run_dir)
        points = []
        for run in summary.get("runs", []):
            conc = run.get("concurrency")
            point = {
                "concurrency": conc,
                "status": run.get("status"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                **{f"{k}_{p}": v for k, vals in profile.get(conc, {}).items()
                   for p, v in vals.items()},
            }
            for key in ("ttft_mean_ms", "itl_mean_ms", "token_throughput",
                        "request_throughput"):
                try:
                    point[key] = float(run[key])
                except (TypeError, ValueError, KeyError):
                    point[key] = None
            points.append(point)
        runs.append({
            "run_id": config.get("run_id"),
            "run_shape": config.get("run_shape"),
            "timestamp_utc": config.get("timestamp_utc"),
            "mode": config.get("mode"),
            "isl": config.get("isl"),
            "osl": config.get("osl"),
            "sweep": config.get("concurrencies"),
            "duration_seconds": config.get("benchmark_duration_seconds"),
            "prefix_reuse_percent": config.get("prefix_reuse_percent"),
            "dataset_file": config.get("dataset_file"),
            "points": points,
        })
    with open(REGISTRY_PATH, "w") as f:
        for run in runs:
            f.write(json.dumps(run) + "\n")
    print(f"registry: {len(runs)} runs -> {REGISTRY_PATH}")


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        print("registry missing; run: uv run python3 scripts/analyze.py registry", file=sys.stderr)
        sys.exit(1)
    return [json.loads(line) for line in REGISTRY_PATH.read_text().splitlines() if line.strip()]


def find_run(runs: list[dict], run_id: str) -> dict | None:
    for run in runs:
        if run["run_id"] == run_id:
            return run
    return None


def numeric(points: list[dict], key: str) -> tuple[list, list]:
    pts = [p for p in points if p.get("status") == "PASS" and isinstance(p.get(key), (int, float))]
    pts.sort(key=lambda p: p["concurrency"])
    return [p["concurrency"] for p in pts], [p[key] for p in pts]


def plot_ablation_throughput(ax, runs: list[dict]) -> bool:
    series = {}
    for phase in ("on", "off"):
        run = find_run(runs, f"ablation-dspark-{phase}")
        if run:
            series[phase] = numeric(run["points"], "tput_p50")
    if "on" not in series or "off" not in series:
        return False
    for phase in ("off", "on"):
        x, y = series[phase]
        ax.plot(x, y, marker="o", color=COLOR[phase], label=LABEL[phase])
    common = sorted(set(series["on"][0]) & set(series["off"][0]))
    for conc in common:
        a = series["on"][1][series["on"][0].index(conc)]
        b = series["off"][1][series["off"][0].index(conc)]
        if a > 0 and b > 0:
            ax.annotate(f"{a / b:.2f}x", (conc, a), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color="#1d4ed8")
    ax.set_xlabel("Concurrency (simultaneous client requests)")
    ax.set_ylabel(f"Output token throughput ({UNITS['tput']})")
    ax.set_title("Serving throughput: DSpark speculative decoding vs non-speculative baseline")
    ax.set_xticks(sorted(set(series["on"][0]) | set(series["off"][0])))
    ax.legend(loc="upper left")
    return True


def plot_ablation_latency(axs, runs: list[dict]) -> bool:
    ax = axs[0]
    drew = False
    for phase in ("off", "on"):
        run = find_run(runs, f"ablation-dspark-{phase}")
        if not run:
            continue
        for key, pct in (("ttft_p50", "p50"), ("ttft_p99", "p99")):
            x, y = numeric(run["points"], key)
            if x:
                ax.plot(x, y, marker="o", color=COLOR[phase],
                        linestyle="-" if pct == "p50" else "--",
                        label=f"{LABEL[phase]} — {pct}")
                drew = True
    ax.set_yscale("log")
    ax.set_xlabel("Concurrency (simultaneous client requests)")
    ax.set_ylabel("Time to first token (ms, log scale)")
    ax.set_title("TTFT percentiles across the saturation knee")
    if drew:
        ax.legend(loc="upper left", fontsize=8)

    ax = axs[1]
    drew2 = False
    for phase in ("off", "on"):
        run = find_run(runs, f"ablation-dspark-{phase}")
        if not run:
            continue
        x, y = numeric(run["points"], "itl_mean")
        if x:
            ax.plot(x, y, marker="o", color=COLOR[phase], label=LABEL[phase])
            drew2 = True
    ax.set_xlabel("Concurrency (simultaneous client requests)")
    ax.set_ylabel("Inter-token latency (ms, mean)")
    ax.set_title("Per-token generation latency")
    if drew2:
        ax.legend(loc="upper left")
    return drew or drew2


def load_spec_series(run_dir: Path) -> dict[str, list]:
    records: dict[int, dict[str, float]] = {}
    tsv = run_dir / "spec-metrics.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            ts, metric, value = parts
            if metric not in ("sglang:spec_accept_length", "sglang:spec_accept_rate"):
                continue
            try:
                records.setdefault(int(ts), {})[metric] = float(value)
            except ValueError:
                continue
    out: dict[str, list] = {"ts": [], "accept_length": [], "accept_rate": []}
    for ts in sorted(records):
        rec = records[ts]
        if "sglang:spec_accept_length" not in rec:
            continue
        out["ts"].append(ts)
        out["accept_length"].append(rec["sglang:spec_accept_length"])
        out["accept_rate"].append(rec.get("sglang:spec_accept_rate"))
    return out


def spec_by_concurrency(run: dict, series: dict) -> tuple[list, list, list]:
    spans = []
    cursor = series["ts"][0] if series["ts"] else 0
    for p in sorted(run["points"], key=lambda p: p["concurrency"] or 0):
        start = cursor + 60
        end = start + int(p.get("elapsed_seconds") or 0)
        spans.append((p["concurrency"], start, end))
        cursor = end + 45
    out_conc, out_len, out_rate = [], [], []
    for conc, start, end in spans:
        vals = [l for t, l in zip(series["ts"], series["accept_length"]) if start <= t <= end]
        rates = [r for t, r in zip(series["ts"], series["accept_rate"])
                 if start <= t <= end and r is not None]
        if vals:
            out_conc.append(conc)
            out_len.append(sum(vals) / len(vals))
            out_rate.append(sum(rates) / len(rates) if rates else None)
    return out_conc, out_len, out_rate


def plot_acceptance(axs, runs: list[dict]) -> bool:
    run = find_run(runs, "ablation-dspark-on")
    if not run:
        return False
    series = load_spec_series(BENCH_ROOT / run["run_id"])
    if not series["accept_length"]:
        return False
    ts0 = series["ts"][0]
    minutes = [(t - ts0) / 60 for t in series["ts"]]

    ax = axs[0]
    ax.plot(minutes, series["accept_length"], color="#1d4ed8", linewidth=1.0)
    ax.axhline(GAMMA, color="#dc2626", linestyle="--", linewidth=1.2,
               label=f"Draft block size γ = {GAMMA}")
    ax.set_xlabel("Time since benchmark start (minutes)")
    ax.set_ylabel("Accepted tokens per verify step")
    ax.set_title("DSpark acceptance length under benchmark load")
    ax.legend(loc="lower right")

    ax = axs[1]
    conc, lengths, rates = spec_by_concurrency(run, series)
    if conc:
        ax.bar([str(c) for c in conc], lengths, color="#1d4ed8", alpha=0.75, width=0.55)
        for i, v in enumerate(lengths):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_ylim(0, GAMMA * 1.15)
    ax.set_xlabel("Concurrency point")
    ax.set_ylabel("Mean accepted tokens per verify step")
    ax.set_title("Acceptance length vs concurrency")
    return True


def plot_real_workload(axs, runs: list[dict]) -> bool:
    real = [r for r in runs if r.get("mode") == "real" and r["points"]]
    if not real:
        return False
    palette = ["#1d4ed8", "#059669", "#b45309", "#7c3aed"]
    ax = axs[0]
    drew = False
    for i, run in enumerate(real):
        x, y = numeric(run["points"], "tput_p50")
        if x:
            ax.plot(x, y, marker="o", color=palette[i % len(palette)], label=run["run_id"])
            drew = True
    ax.set_xlabel("Concurrency (simultaneous client requests)")
    ax.set_ylabel(f"Output token throughput ({UNITS['tput']})")
    ax.set_title("Real agentic-coding prompts: throughput")
    if drew:
        ax.legend(loc="upper left", fontsize=8)

    ax = axs[1]
    drew2 = False
    for i, run in enumerate(real):
        x, y = numeric(run["points"], "ttft_p50")
        if x:
            ax.plot(x, y, marker="o", color=palette[i % len(palette)], label=run["run_id"])
            drew2 = True
    ax.set_yscale("log")
    ax.set_xlabel("Concurrency (simultaneous client requests)")
    ax.set_ylabel("Time to first token (ms, log scale)")
    ax.set_title("Real agentic-coding prompts: TTFT (multi-turn sessions)")
    if drew2:
        ax.legend(loc="upper left", fontsize=8)
    return drew or drew2


def fmt(v) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    return f"{v:,.1f}"


def run_table(run: dict, headers: tuple) -> list[str]:
    cols = {
        "throughput": ("Throughput (tok/s)", "tput_p50"),
        "ttft_p50": ("TTFT p50 (ms)", "ttft_p50"),
        "ttft_p99": ("TTFT p99 (ms)", "ttft_p99"),
        "itl": ("ITL mean (ms)", "itl_mean"),
    }
    head = "| Concurrency | " + " | ".join(cols[c][0] for c in headers) + " |"
    sep = "|---:|" + "---:|" * len(headers)
    rows = [head, sep]
    for p in sorted(run["points"], key=lambda p: p["concurrency"] or 0):
        rows.append(f"| {p['concurrency']} | "
                    + " | ".join(fmt(p.get(cols[c][1])) for c in headers) + " |")
    return rows


def report() -> None:
    style()
    runs = load_registry()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    if plot_ablation_throughput(ax, runs):
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "ablation-throughput.png")
        figures.append("ablation-throughput.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    if plot_ablation_latency(axs, runs):
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "ablation-latency.png")
        figures.append("ablation-latency.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
    if plot_acceptance(axs, runs):
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "acceptance.png")
        figures.append("acceptance.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    if plot_real_workload(axs, runs):
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "real-workload.png")
        figures.append("real-workload.png")
    plt.close(fig)

    lines = ["# DSpark Serving — Experiment Report", ""]
    for run_id in ("ablation-dspark-on", "ablation-dspark-off"):
        run = find_run(runs, run_id)
        if not run:
            continue
        lines += [f"## {run['run_id']}", "",
                  f"Mode `{run['mode']}`, ISL {run['isl']}, OSL {run['osl']}, "
                  f"{run['duration_seconds']}s per point, sweep {run['sweep']}.", ""]
        lines += run_table(run, ("throughput", "ttft_p50", "ttft_p99", "itl"))
        lines.append("")
    for run in runs:
        if run.get("mode") == "real":
            lines += [f"## {run['run_id']}", "",
                      f"Real-prompt workload ({run.get('dataset_file') or 'custom dataset'}), "
                      f"{run['duration_seconds']}s per point, sweep {run['sweep']}.", ""]
            lines += run_table(run, ("throughput", "ttft_p50", "ttft_p99"))
            lines.append("")
    if figures:
        lines += ["## Figures", ""]
        for name in figures:
            lines += [f"![{name}]({name})", ""]
    (REPORT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"report: {len(figures)} figures + REPORT.md -> {REPORT_DIR}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "registry"
    if cmd == "registry":
        registry()
    elif cmd == "report":
        report()
    else:
        print(f"unknown subcommand: {cmd} (use registry|report)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

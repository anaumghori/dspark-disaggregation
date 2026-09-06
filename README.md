# DSpark Speculative-Decoding Serving Engine

```THIS IS STILL UNDER TESTING TO BE MORE ROBUST```

This repository deploys and benchmarks the DeepSeek-V4-Flash model with DSpark speculative decoding using four NVIDIA RTX Pro 6000 Blackwell GPUs, with Dynamo orchestrating the deployment and SGLang acting as the serving engine. The goal of the project is twofold: to run a production-style disaggregated inference deployment, and to characterize whether DSpark speculative decoding actually helps on real hardware by running a controlled on/off measurement study against a real prompt workload.

At a high level, the deployment starts a small fleet of processes: a Dynamo frontend accepts OpenAI-compatible chat requests on HTTP port 8000 and routes them to a disaggregated engine pair. A prefill worker (GPUs 0-1) reads incoming prompts and computes their KV cache, then transfers that cache over the NIXL transfer backend to a decode worker (GPUs 2-3) which generates the output tokens. The decode worker runs the DSpark speculative decoding algorithm. Because the deployment is disaggregated, the two workers run as separate SGLang engine processes kept in sync by the Dynamo frontend.

This project introduces two operating modes:

- **Experiment Mode** — runs the full automated measurement study and produces a report with plots.
- **Interactive Mode** — starts the serving stack in the foreground so it can be exercised directly, smoke-tested, and monitored live.

### Repository layout

```
.
├── Makefile                     # Every top-level operation as a make target
├── configs/
│   ├── prefill.args             # Launch flags for the prefill worker
│   └── decode.args              # Launch flags for the decode worker (DSpark flags live here)
├── deploy/
│   ├── launch/
│   │   ├── serve.sh             # Starts frontend + prefill + decode workers
│   │   └── stop.sh              # Stops the processes started by serve.sh
│   └── common/
│       ├── launch_utils.sh      # Shared launch helpers (.env overlay, args files, logging)
│       ├── output.sh            # Shared console output helpers (colors, log, pass/fail)
│       └── gpu_utils.sh         # Shared GPU inventory/memory helpers
└── scripts/
    ├── setup_host.sh            # One-time system dependency installation
    ├── m0_manifest.sh           # Environment health assertions (make manifest)
    ├── m1_sm120_parity.py       # Numerical parity check for the sm120 kernels (make parity)
    ├── calibrate_sps_sts.sh     # Generates the DSpark calibration tables (make calibrate)
    ├── build_dataset.py         # Downloads + samples the real-prompt dataset (make dataset)
    ├── ablation.sh              # Runs the full experiment mode study (make ablation)
    ├── benchmark.sh             # Measurement engine called by the ablation script
    ├── analyze.py               # Builds the results registry and report figures
    ├── smoke_test.sh            # Interactive live correctness check (make smoke-test)
    ├── start_monitoring.sh      # Starts Prometheus + Grafana (make monitoring)
    └── stop_monitoring.sh       # Stops Prometheus + Grafana (make stop-monitoring)
```

<br><br>

## One-time setup and calibration (required for both modes)

Steps 1–4 configure the machine once per model and GPU combination. They apply regardless of which mode you run and are not repeated on every session. The DSpark decode worker needs a measured step-cost table before it will start. This is a real measurement taken from your GPUs, so it cannot be shipped with the repository. Steps 5–6 are optional health checks that can be re-run at any time in either mode.

| Step | Command | Purpose | Required |
|-----|---------|---------|----------|
| 1 | `git clone <this-repo> && cd speculative-decoding-engine` | Place the repository on the GPU machine and enter it | Required |
| 2 | `make setup` | Install the system toolchain, create the pinned Python environment, run the health gate | Required |
| 3 | `make calibrate A='sps'` | Generate the SPS step-cost table into `configs/dspark_sps_table.json` | Required |
| 4 | `make calibrate A='sts'` | Generate the STS confidence-tuning table into `configs/dspark_sts_table.json` | Optional |
| 5 | `make manifest` | Re-run the environment health check as an ad-hoc verification | Optional |
| 6 | `make parity` | Verify the fast GPU kernels produce the same answers as the reference implementations | Optional |

The `Required` column indicates whether the step must be completed before serving. The SPS table (step 3) is required; the decode worker will not start without it. The STS table (step 4), `make manifest` (step 5), and `make parity` (step 6) are optional and can be skipped.

<br><br>

## Mode 1 — Experiment Mode (automated measurement study)

Experiment Mode runs the serving stack twice against the same real prompt workload, once with DSpark speculative decoding enabled and once with it stripped out, while sampling the decode worker's speculative metrics, then automatically produces a results registry, four comparison plots, and a markdown report. It is a single self-contained command that manages its own server lifecycles, so you do not start or stop servers yourself.

**Starting:** Complete `One-time setup and calibration` steps 1–3 first, then follow the mode-specific commands below:

| Step | Command | Purpose |
|-----|---------|---------|
| 1 | `make dataset` | Build the real agentic-coding prompt dataset (once) |
| 2 | `make ablation` | Run the full on/off study, acceptance sampling, and report generation (starts/stops servers itself) |
| 3 | `make stop` | Only if you want to interrupt `make ablation` early: stops any still-running deployment (otherwise `make ablation` stops itself automatically when completed) |

Experiment Mode measures standard serving latency and throughput per concurrency level: time to first token (TTFT), inter-token latency (ITL), output token throughput, and request throughput, including p50 and p99 percentiles, using the AIPerf benchmarking client against the real prompts. In addition, it records the DSpark acceptance length (accepted tokens per verify step) and draft acceptance rate from the decode worker, which are the quantities that determine whether speculative decoding is actually saving work. The four figures produced are throughput with spec on versus off (with the speedup ratio annotated at each concurrency point), TTFT and ITL on versus off, the acceptance-length time series with the gamma reference line plus mean acceptance per concurrency point, and the real-prompt workload panels. Results and figures are written under `benchmarks/`.

```
benchmarks/
├── datasets/
│   ├── agentic_code22.jsonl      # the real prompts sent by the benchmark client
│   └── dataset-stats.json        # token distribution and sampling summary
├── ablation-dspark-on/           # phase 1 artifacts (spec decoding on)
│   ├── run-config.json           # immutable record of the run settings
│   ├── gpu-snapshots.csv         # GPU state before and after the phase
│   ├── spec-metrics.tsv          # five-second acceptance samples (on phase only)
│   ├── sweep-summary.json        # per-concurrency results (analysis input)
│   ├── worker-metrics-after.prefill.prom and .decode.prom
│   └── c<concurrency>/           # one dir per point (AIPerf export plus error log)
├── ablation-dspark-off/          # phase 2 artifacts (spec decoding off)
├── experiments.jsonl             # merged results registry
└── report/
    ├── REPORT.md                 # the written results report
    └── *.png                     # the four comparison plots
```

<br><br>

## Mode 2 — Interactive Mode (manual serving and live inspection)

Interactive Mode is for the cases where you simply want the serving stack up so you can use it or watch it directly: validate the deployment is healthy, point an OpenAI-compatible client at it, exercise it with live traffic, or watch its metrics on a dashboard. In this mode you start and stop the servers yourself, and you must run the commands in the order shown because each later step requires an already-running server. Any OpenAI-compatible client pointed at `http://localhost:8000` (base URL `http://<host>:8000/v1`, model `deepseek-v4-flash-dspark`) can be used against the same endpoint, including curl, the OpenAI SDK, or CLI tools that let you configure a custom base URL and model name.

**Starting:** Complete `One-time setup and calibration` steps 1–3 first. Then start the serving stack directly:

| Step | Command | Purpose |
|-----|---------|---------|
| 1 | `make serve` | Start the frontend plus the prefill and decode workers in the foreground |
| 2 | `make smoke-test` | Verify the running deployment with a real end-to-end chat request |
| 3 | `make monitoring` | Optional: start Prometheus and Grafana to visualize metrics live |
| 4 | `make stop-monitoring` | Only if you started monitoring: stop Prometheus and Grafana |
| 5 | `make stop` | Required: stop the serving stack yourself and release the four GPUs (you started it, it does not stop on its own) |

`make serve` stays in the foreground and logs progress while the model loads into the GPUs over the first few minutes. Once the log settles, run `make smoke-test` to confirm the deployment is operational. Always run `make stop` before shutting the machine down so the GPU workers are torn down and you do not keep paying for reserved GPU time.

<br><br>

## Reference: make commands

#### `make setup`
The one-time machine bring-up. It runs `scripts/setup_host.sh`, which installs the system build dependencies (libraries, Rust toolchain, protobuf) needed to compile the pinned dependencies from source, then `uv sync` to create the pinned Python environment from `pyproject.toml` containing SGLang, Dynamo, NIXL, and matplotlib, and finally `scripts/m0_manifest.sh` as a health gate that verifies all four GPUs and the required software. Every step is idempotent, so re-running it skips completed work and is safe.

#### `make manifest`
Runs `scripts/m0_manifest.sh` alone as a standalone environment health check. It asserts the pinned dependency versions, the import surface, CUDA device and peer access, the kernel-cache directories, and, once the model is downloaded, checkpoint facts. It is an optional verification you can run any time to confirm the machine is still configured correctly.

#### `make parity`
Runs `scripts/m1_sm120_parity.py`, a numerical parity sweep over the DSpark kernel family. It compares the fast Triton-based GPU kernels against the slow reference implementations and prints a pass or fail line for each one, confirming that the hardware-specific code paths compute correct answers. It is optional and not required before serving, but it is a cheap way to catch kernel-level problems after setup or when a suspicion arises.

#### `make calibrate A='sps'`
Runs the `sps` subcommand of `scripts/calibrate_sps_sts.sh`. It starts a dedicated temporary SGLang server on port 30000 in a static verification mode that records step timings, waits for it to be ready, runs the DSpark step-cost profiler across batch sizes, writes the result to `configs/dspark_sps_table.json`, and then stops the server. This SPS table is the required input for the DSpark budget planner, and without it the decode worker refuses to start, so it must be generated once per model and GPU combination. The first run downloads the model checkpoint and takes a long while, but later runs reuse the download, and once generated the table can be committed to git and skipped on identical hardware.

#### `make calibrate A='sts'`
Runs the `sts` subcommand of the same calibration script. It starts a temporary server on port 30001 in compact verification mode that records raw DSpark confidence data, tells you to drive traffic against it, and then, when you stop the server with Ctrl+C, fits the STS confidence calibration from the collected shards and writes `configs/dspark_sts_table.json`. This calibration is optional: the launch scripts remove its flag when the table is absent and the engine runs without it, so you can skip it if you do not need confidence tuning.

#### `make dataset`
Runs `scripts/build_dataset.py`, which downloads the public `novita/agentic_code_dataset_22` corpus (22 recorded agentic coding sessions) once into `.cache/datasets/`, samples 96 contiguous multi-turn conversation windows of roughly 2,000 to 24,000 estimated tokens each using a deterministic seed, and writes them as an AIPerf multi-turn dataset to `benchmarks/datasets/agentic_code22.jsonl` together with a `dataset-stats.json` manifest. It requires no GPU and is the prerequisite for Experiment Mode only; Interactive Mode never uses it.

#### `make ablation`
Runs `scripts/ablation.sh`, the Experiment Mode orchestrator. It protects itself with dataset preconditions (it only accepts the real dataset and errors out if the file is missing), then executes two phases automatically. The first phase launches the deployment with speculative decoding enabled via `serve.sh`, waits for the frontend and both workers to be ready, and verifies that DSpark metrics are actually present, failing if anything is off. While it runs it polls the decode worker every five seconds and records the DSpark acceptance metrics to `spec-metrics.tsv`. It then benchmarks the running server across six concurrency levels (1, 4, 16, 32, 64, 128) for 120 seconds each using the real dataset, and finally stops the server itself. The second phase repeats the whole sequence with speculative decoding stripped from the decode launch. It manages server start and stop itself via `deploy/launch/serve.sh` and `deploy/launch/stop.sh`, calls `scripts/benchmark.sh` as its measurement engine for each phase, and when both phases finish it automatically rebuilds the results registry, renders the four plots, writes `REPORT.md`, and prunes redundant raw artifacts. This is the single command that produces all of the project's experimental results. 

#### `make smoke-test`
Runs `scripts/smoke_test.sh`, which validates an already-running deployment with a real end-to-end request. It checks that the frontend endpoints respond, that the model is registered, that a chat completion request returns a valid response containing the expected content within a time bound, and that both workers expose healthy metric endpoints with DSpark speculative metrics present. It prints a pass or fail line for each check and returns a nonzero exit code if any check fails. It requires the server to already be up, as started by `make serve`, so it belongs to Interactive Mode.

#### `make monitoring` / `make stop-monitoring`
Run `scripts/start_monitoring.sh` and `scripts/stop_monitoring.sh`. The former launches a local Prometheus instance that scrapes the workers' metric endpoints and a Grafana instance that renders them on a pre-loaded DSpark dashboard, using the `prometheus` and `grafana` binaries on the machine. After starting, Prometheus is available at http://localhost:9090 and Grafana at http://localhost:3000 with the default login admin/admin. Monitoring is optional and does not alter the serving stack; use it when you want to watch a live run, and stop it explicitly with `make stop-monitoring` when you are done so the processes do not keep running.

#### `make serve`
Runs `deploy/launch/serve.sh`, which starts the Dynamo frontend (HTTP port 8000), the prefill worker (GPUs 0-1, system status port 8081), and the decode worker running DSpark (GPUs 2-3, system status port 8082) in the foreground. It reads the worker launch arguments from `configs/prefill.args` and `configs/decode.args`, wires the kernel cache directories, and requires the SPS table to be present, returning a fatal error otherwise. It is the entry point for Interactive Mode and is also the launch script that Experiment Mode drives internally.

#### `make stop`
Runs `deploy/launch/stop.sh`, which terminates exactly the processes recorded by the most recent launch (the frontend and both workers) and releases all four GPUs. It is the correct teardown for both modes, and running it before a fresh `make serve` or `make ablation` always frees the GPUs. Run it before shutting the machine down to avoid paying for reserved GPU time.
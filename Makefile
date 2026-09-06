PYTHON := python3

.PHONY: setup manifest parity calibrate serve stop smoke-test dataset ablation monitoring stop-monitoring

setup:
	./scripts/setup_host.sh
	uv sync
	uv run ./scripts/apply_sm120_patches.sh
	uv run ./scripts/m0_manifest.sh

manifest:
	uv run ./scripts/m0_manifest.sh

parity:
	uv run $(PYTHON) scripts/m1_sm120_parity.py

calibrate:
	uv run ./scripts/calibrate_sps_sts.sh $(A)

serve:
	./deploy/launch/serve.sh

smoke-test:
	./scripts/smoke_test.sh

dataset:
	uv run python3 scripts/build_dataset.py $(DATASET_ARGS)

ablation:
	uv run ./scripts/ablation.sh

stop:
	./deploy/launch/stop.sh

monitoring:
	./scripts/start_monitoring.sh

stop-monitoring:
	./scripts/stop_monitoring.sh

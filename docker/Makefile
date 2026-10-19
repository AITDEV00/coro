# coro docker build & run orchestration.
#
# Wraps docker/Dockerfile (the slim multi-stage runtime image) so the two
# documented flavours are one command each, with the upstream defaults kept in
# sync with .github/workflows -> docker/ci-docker.yml and README.md. The image
# build context stays the repo root (`context: .`) because the Dockerfile COPYs
# pyproject.toml / uv.lock / coro/ from it — run make from anywhere, the paths
# are anchored:  make -C docker <target>  works too.
#
# Flavours (mirroring the pyproject extras, mutually exclusive):
#   cpu  EXTRA=cpu  CORE_IMAGE=ubuntu:noble
#   gpu  EXTRA=cuda CORE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04
#
# speakrs backend users: the wheel is not on PyPI (see docs/adr/0025). Build it
# from avencera/speakrs, drop it into docker/wheels/, and it is baked in
# automatically; see `make help` in this folder.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- image identity -----------------------------------------------------------
IMAGE_NAME ?= coro
REGISTRY ?= ghcr.io/collectiveai-team
# PEP 440 version; CI derives it from git via hatch-vcs, local builds default.
CORO_VERSION ?= 0.0.0

# --- flavour knobs (override on the command line) ----------------------------
#   make gpu                         -> gpu flavour, tag coro:gpu
#   make build FLAVOUR=gpu IMAGE=me/coro:tag
FLAVOUR ?= cpu

ifeq ($(FLAVOUR),gpu)
EXTRA := cuda
CORE_IMAGE := nvidia/cuda:13.0.3-base-ubuntu24.04
else
EXTRA := cpu
CORE_IMAGE := ubuntu:noble
endif

TAG ?= $(IMAGE_NAME):$(FLAVOUR)

# --- speakrs wheel (optional; docker/wheels/*.whl is gitignored) ----------------
# The wildcard runs in this makefile's directory (docker/), while the build
# context is the repo root, so the build-arg path is prefixed accordingly.
# Drop the maturin-built wheel into docker/wheels/ and it is picked up
# automatically; or pass SPEAKRS_WHEEL=<path-from-repo-root> explicitly.
# (maturin's real tag is manylinux_2_35_x86_64, so match any speakrs wheel;
# firstword keeps the build-arg a single path if several are staged.)
SPEAKRS_WHEEL_FILE ?= $(firstword $(wildcard wheels/speakrs-*.whl))
ifneq ($(SPEAKRS_WHEEL_FILE),)
SPEAKRS_BUILD_ARG := --build-arg SPEAKRS_WHEEL=docker/$(SPEAKRS_WHEEL_FILE)
endif

BUILDX_FLAGS := --build-arg CORE_IMAGE=$(CORE_IMAGE) \
                --build-arg EXTRA=$(EXTRA) \
                --build-arg CORO_VERSION=$(CORO_VERSION) \
                $(SPEAKRS_BUILD_ARG)

# HF cache volume for model weights (speakrs: ~31 MB; pyannote/nemo larger).
HF_CACHE_VOLUME ?= coro-hf-cache
PORT ?= 8000

# --- ADEO cluster registry (omnivoice-server convention) -----------------------
# Same Harbor registry as the omnivoice internal images; the coro speakrs
# image lives under the mlops-serving namespace.
ADEO_REGISTRY ?= registry.adeoaiengine.ecouncil.ae
ADEO_REGISTRY_PATH ?= openinnovationai/platform/mlops/mlops-serving
ADEO_IMAGE ?= $(ADEO_REGISTRY)/$(ADEO_REGISTRY_PATH)/coro
ADEO_TAG ?= $(ADEO_IMAGE):speakrs
ADEO_PORT ?= 8080
# Harbor robot account for the mlops-serving namespace. Defaults make `make
# login` non-interactive; override either on the command line for other
# accounts:  make login ADEO_REGISTRY_USER=... ADEO_REGISTRY_PASSWORD=...
ADEO_REGISTRY_USER ?= custom-images
ADEO_REGISTRY_PASSWORD ?= CustomImages123

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make [FLAVOUR=cpu|gpu] <target>\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build: ## Build the image (FLAVOUR=cpu|gpu; TAG=... to override the tag).
	docker build -f docker/Dockerfile -t $(TAG) $(BUILDX_FLAGS) ..

.PHONY: cpu
cpu: ## Build the CPU flavour (tag coro:cpu).
	$(MAKE) build FLAVOUR=cpu TAG=$(IMAGE_NAME):cpu

.PHONY: gpu
gpu: ## Build the GPU flavour (tag coro:gpu; needs NVIDIA Container Toolkit only to run).
	$(MAKE) build FLAVOUR=gpu TAG=$(IMAGE_NAME):gpu

.PHONY: run-cpu
run-cpu: ## Run the CPU image on $(PORT) (backend defaults; mount HF cache volume).
	docker run --rm -p $(PORT):8000 -v $(HF_CACHE_VOLUME):/root/.cache/huggingface \
		$(IMAGE_NAME):cpu serve --port $(PORT)

.PHONY: run-gpu
run-gpu: ## Run the GPU image with --gpus all on $(PORT).
	docker run --rm --gpus all -p $(PORT):8000 -v $(HF_CACHE_VOLUME):/root/.cache/huggingface \
		$(IMAGE_NAME):gpu serve --port $(PORT)

.PHONY: run-speakrs
run-speakrs: ## Run the GPU image with the speakrs diarization backend (cuda-fast, 4 workers).
	docker run --rm --gpus all -p $(PORT):8000 -v $(HF_CACHE_VOLUME):/root/.cache/huggingface \
		$(IMAGE_NAME):gpu serve --port $(PORT) --backend-diarization speakrs

.PHONY: run-speakrs-cuda
run-speakrs-cuda: ## Run the speakrs backend on the accuracy tier (CORO_DIARIZATION_SPEAKRS_MODE=cuda).
	docker run --rm --gpus all -p $(PORT):8000 -v $(HF_CACHE_VOLUME):/root/.cache/huggingface \
		-e CORO_DIARIZATION_SPEAKRS_MODE=cuda \
		$(IMAGE_NAME):gpu serve --port $(PORT) --backend-diarization speakrs

.PHONY: login
login: ## Log in to the ADEO Harbor registry (non-interactive; ADEO_REGISTRY_USER/PASSWORD defaults).
	@printf '%s' '$(ADEO_REGISTRY_PASSWORD)' | docker login $(ADEO_REGISTRY) -u '$(ADEO_REGISTRY_USER)' --password-stdin

.PHONY: fetch-models
fetch-models: ## Export the pinned speakrs weights from the HF cache into docker/models/speakrs/ (run once after a local speakrs run).
	SNAP=$$(ls -d ~/.cache/huggingface/hub/models--avencera--speakrs-models/snapshots/*/ 2>/dev/null | head -1); \
	if [ -z "$$SNAP" ]; then \
		echo "No speakrs snapshot in the HF cache yet; run the engine once locally (it downloads 31 MB) or use huggingface-cli."; exit 1; \
	fi; \
	mkdir -p models/speakrs; \
	for f in segmentation-3.0.safetensors wespeaker-multimask-tail.safetensors \
		 wespeaker-voxceleb-resnet34.min_num_samples.txt \
		 plda_lda.npy plda_tr.npy plda_mu.npy plda_psi.npy plda_mean1.npy plda_mean2.npy; do \
		cp -L "$$SNAP/$$f" models/speakrs/$$f; \
	done; \
	echo "Exported 9 speakrs files ($$(du -sh models/speakrs | cut -f1)) — commit NOT required (gitignored)."

.PHONY: fetch-canary
fetch-canary: ## Export the canary ASR snapshot (snapshots/ + refs/) from the HF cache into docker/models/canary/.
	CANARY_SNAP_DIR=$$HOME/.cache/huggingface/hub/models--collectiveai--canary-1b-v2-onnx-split-int8; \
	if [ ! -d "$$CANARY_SNAP_DIR/snapshots" ]; then \
		echo "No canary snapshot in the HF cache; run coro once locally (it downloads ~1.3 GB) or use huggingface-cli."; exit 1; \
	fi; \
	rev=$$(cat "$$CANARY_SNAP_DIR/refs/main"); \
	mkdir -p models/canary/snapshots/$$rev models/canary/refs; \
	for f in config.json vocab.txt \
			 encoder-model.static_qdq_v4_pct_excl.onnx encoder-model.static_qdq_v4_pct_excl.onnx.data \
			 decoder_step.dynamic_v1_quint8.onnx xattn_kv.onnx; do \
		cp -L "$$CANARY_SNAP_DIR/snapshots/$$rev/$$f" models/canary/snapshots/$$rev/$$f; \
	done; \
	cp "$$CANARY_SNAP_DIR/refs/main" models/canary/refs/main; \
	echo "Exported canary snapshot $$rev ($$(du -sh models/canary | cut -f1)) — commit NOT required (gitignored)."

.PHONY: adeo
adeo: ## Build the ADEO cluster image (port 8080, UID 10000, baked weights, /health-check).
	docker build -f docker/Dockerfile.adeo -t $(ADEO_TAG) $(SPEAKRS_BUILD_ARG) ..

.PHONY: adeo-push
adeo-push: ## Push the ADEO image to the Harbor registry (requires make login first).
	docker push $(ADEO_TAG)

.PHONY: run-adeo
run-adeo: ## Run the ADEO image locally on $(ADEO_PORT) (read-only root, tmpfs /tmp, --gpus all).
	docker run --rm --gpus all -p $(ADEO_PORT):8080 --user 10000:10000 \
		--read-only --tmpfs /tmp:rw,size=8g $(ADEO_TAG)

.PHONY: push
push: ## Push TAG to REGISTRY (or pass IMAGE=registry/repo:tag).
	docker tag $(TAG) $(REGISTRY)/$(TAG) && docker push $(REGISTRY)/$(TAG)

.PHONY: clean
clean: ## Remove local coro images and the HF cache volume.
	docker rmi $(IMAGE_NAME):cpu $(IMAGE_NAME):gpu 2>/dev/null; \
	docker volume rm $(HF_CACHE_VOLUME) 2>/dev/null; true

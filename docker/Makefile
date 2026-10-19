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
SPEAKRS_WHEEL_FILE ?= $(wildcard wheels/speakrs-*-linux_x86_64.whl)
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

.PHONY: push
push: ## Push TAG to REGISTRY (or pass IMAGE=registry/repo:tag).
	docker tag $(TAG) $(REGISTRY)/$(TAG) && docker push $(REGISTRY)/$(TAG)

.PHONY: clean
clean: ## Remove local coro images and the HF cache volume.
	docker rmi $(IMAGE_NAME):cpu $(IMAGE_NAME):gpu 2>/dev/null; \
	docker volume rm $(HF_CACHE_VOLUME) 2>/dev/null; true

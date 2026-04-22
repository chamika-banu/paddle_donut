.PHONY: build build-gpu build-prod build-prod-gpu \
        serve serve-build serve-gpu serve-build-gpu \
        serve-prod serve-prod-gpu

# ── CPU (default) ─────────────────────────────────────────────────────────────
# Works on any machine — no GPU or NVIDIA Container Toolkit required.

build:
	docker compose build

build-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

serve:
	docker compose up

serve-build:
	docker compose up --build

serve-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up

# ── GPU ────────────────────────────────────────────────────────────────────────
# Requires NVIDIA GPU drivers and NVIDIA Container Toolkit on the host.
# Donut (torch) runs on GPU. PaddleOCR remains CPU-only in both images.

build-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build

build-prod-gpu:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.prod-gpu.yml build

serve-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

serve-build-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

serve-prod-gpu:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.prod-gpu.yml up

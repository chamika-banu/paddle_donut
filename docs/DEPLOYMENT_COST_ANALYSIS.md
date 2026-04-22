# Deployment Requirements & Cost Analysis
**Service:** ID Verification Microservice (Tiered Pipeline v2.0)
**Date:** April 2026  
**All prices in USD/month at 24/7 continuous running unless stated otherwise**

---

## 0. Tiered Pipeline Cost Impact (v2.0)

**This section reflects the cost implications of the tiered architecture introduced in v2.0.**

### What changed

**Startup grace period:**
- CPU: ~5–15s (Donut is ~500 MB). Set an initial delay of **30 seconds** to be safe.
- GPU: ~3–6s. **15 seconds** initial delay is sufficient.

The v2.0 tiered pipeline replaced Florence-2 with **Donut** (~500 MB) and introduced escalation logic:

- **Tier 1 (PaddleOCR + regex)** handles ~70–90% of clean IDs with no VLM invocation — these requests take < 2s on any hardware.
- **Tier 2 (Donut DocVQA)** is invoked only for ambiguous/difficult IDs (~10–30% of requests).

### Cost implications

| Metric | v1.x (always-on Florence-2) | v2.0 (tiered Donut pipeline) |
|---|---|---|
| VLM invocations | 100% of requests | ~10–30% of requests |
| VLM model size | 1.54 GB (Florence-2-large-ft) | ~500 MB (Donut) |
| RAM floor | ~12–14 GB | ~3.5–5 GB |
| Tier 1 request time (CPU) | N/A | < 2s |
| Tier 2 request time (CPU) | 10–30s (always) | 4–10s (escalation only) |
| Tier 2 request time (GPU) | 2–6s (always) | 1–2s (escalation only) |
| Effective avg response time (CPU) | 10–30s | < 3s (weighted average at 20% escalation) |
| Effective avg response time (GPU) | 2–6s | < 1.5s (weighted average at 20% escalation) |

### GPU cost reassessment

At 20% escalation rate, a GPU instance is only running Donut for ~1 in 5 requests. The cost case for GPU is now even more favourable:

- **GPU with tiered pipeline:** ~$0.10–0.20/request at serverless GPU pricing (e.g. RunPod) — Donut at 1–2s and only for 20% of requests.
- **CPU with tiered pipeline:** Viable for very high loads because Donut is ~3× faster than Florence-2 on CPU, and 70–90% of requests return in < 2s.

### Recommendation update

The Donut migration further strengthens the case for **CPU deployment**. Donut's ~500 MB size and faster inference make the CPU-vs-GPU latency gap smaller than with Florence-2. Upgrade to GPU only if you observe sustained > 5s p95 latency in production.

---

## 1. Minimum System Requirements

These are the absolute floor specifications — the service will run but with no headroom for concurrent requests or OS overhead.

### CPU
**4 vCPUs minimum.**  
Donut inference is single-threaded in PyTorch by default, but the OS, uvicorn worker, PaddleOCR, and concurrent asyncio tasks all compete for cores. With fewer than 4 cores the service becomes CPU-starved during inference and uvicorn begins queuing requests.

```bash
# Dev
make serve

# Prod
export DONUT_IMAGE=your-registry/donut:latest
make serve-prod
```

### GPU

```bash
# Dev (GPU)
make serve-gpu

# Prod (GPU)
export DONUT_IMAGE=your-registry/donut:gpu-latest
make serve-prod-gpu
```

### RAM
**8 GB minimum.**  
The memory budget breaks down as follows:

| Component | Memory |
|---|---|
| Donut weights (FP32, CPU) | ~1.0 GB |
| Donut inference peak | ~0.5–1 GB |
| PaddleOCR model weights | ~0.3 GB |
| PaddleOCR inference peak | ~0.5–1 GB |
| OS + uvicorn + FastAPI runtime | ~1–2 GB |
| Image buffer per request (5 MB × 2 temp copies) | ~0.1 GB |
| **Total floor** | **~7–8 GB** |

8 GB provides minimal headroom. 16 GB is recommended for production to handle concurrent requests without OOM risk.

### Storage
**30 GB minimum.**  
The Docker image is a "fat container" that already contains the weights for Donut (~500 MB) and PaddleOCR (~50 MB). The total image size typically reaches 6–8 GB. OS and system layers take ~5 GB. 30 GB provides ample headroom.

### GPU
**Not required.** PaddleOCR runs CPU-only regardless. Donut will fall back to CPU inference automatically. GPU meaningfully reduces Donut inference time (4–10s on CPU → 1–2s on GPU) but is not required for the service to function.

### Network
**100 Mbps minimum.** Image uploads are 2–5 MB. At 100 Mbps a 5 MB image uploads in ~400ms, which is acceptable before inference begins. Inbound traffic is free on all major providers.

---

## 2. Recommended Production Requirements

These specs provide comfortable headroom for concurrent requests, OS overhead, and peak memory during inference without risking OOM termination.

### CPU
**8 vCPUs.**  
Allows Donut and PaddleOCR to run concurrently (via `asyncio.gather` + thread pool) without contention, with cores left for the OS, uvicorn, and request handling. 4 vCPUs is viable for purely sequential single-user load; 8 covers low-to-medium concurrent traffic comfortably.

### RAM
**32 GB.**  
Doubles the headroom above the memory floor. This is the most important spec — memory pressure is the primary failure mode for this service. 32 GB allows two concurrent requests to be in-flight simultaneously without risk of OOM. For a verification microservice called from a NestJS backend, two concurrent requests is a realistic peak.

### Storage
**64 GB SSD.**  
Accommodates model weight caches, Docker image layers, uvicorn logs, and leaves room for OS updates and temporary files without manual cleanup.

### GPU
**Optional but recommended if latency matters.** An NVIDIA T4 (16 GB VRAM) is the minimum GPU tier that can hold Donut comfortably in VRAM with inference headroom. The T4 reduces Donut inference from 4–10s to ~1–2s. PaddleOCR stays on CPU regardless of GPU availability, so total request time on GPU is approximately 1–3s vs 5–12s on CPU.

- [ ] `/health` returns 200 after startup with `"donut_model": "Donut-base (escalation only)"`
- [ ] NestJS timeout is set to 60s (CPU) or 30s (GPU)
- [ ] Named volumes (`donut_hf_cache`, `donut_paddle_cache`) are created or already exist
- [ ] On first startup, internet access is available to download models (~550 MB total)
- [ ] GPU hosts: NVIDIA Container Toolkit is installed
- [ ] If upgrading from Florence-2: clear old `florence2_hf_cache` volume — Donut uses `donut_hf_cache`

For a low-to-medium load verification service where users are waiting for a result, **GPU is worth it** — 1s vs 5s is a meaningful UX difference.

### Network
**1 Gbps.** Standard on all modern cloud instances. No special requirements.

---

## 3. Azure Cost Breakdown

> Prices are Pay-As-You-Go Linux (Ubuntu) in East US region, April 2026. Reserved Instance (1-year) discounts of approximately 35–40% are available for all tiers and noted where relevant. Prices sourced from Vantage and Holori instance trackers against the Azure API.

### Shared Additional Costs (all Azure tiers)

| Item | Monthly Cost |
|---|---|
| Azure Container Registry (Basic tier, optional) | ~$5 |
| Managed OS disk 64 GB (P6 Premium SSD) | ~$10 |
| Outbound egress (first 5 GB free, ~10 GB/mo estimated) | ~$1 |
| Public IP address | ~$3 |
| **Additional costs subtotal** | **~$19/mo** |

---

### Option A — CPU Minimum (`Standard_E4as_v5`)

| Spec | Value |
|---|---|
| vCPUs | 4 |
| RAM | 32 GB |
| Instance family | Eav5 (memory-optimised, AMD EPYC) |
| On-demand price | $0.226/hr |

| Line item | Monthly |
|---|---|
| VM compute (730 hrs) | $165 |
| Additional costs | $19 |
| **Total** | **~$184/mo** |

**1-year Reserved:** ~$107/mo compute + $19 = **~$126/mo**

**Notes:** The E-series is the right family here — 8 GB RAM per vCPU means a 4-vCPU instance gives you 32 GB, which meets the recommended RAM spec while keeping vCPU count at the minimum. The `as` suffix denotes AMD EPYC, which is ~10% cheaper than the Intel equivalent (`Standard_E4s_v5` at $0.252/hr) with comparable performance for this workload.

---

### Option B — CPU Recommended (`Standard_E8as_v5`)

| Spec | Value |
|---|---|
| vCPUs | 8 |
| RAM | 64 GB |
| Instance family | Eav5 (memory-optimised, AMD EPYC) |
| On-demand price | ~$0.452/hr |

| Line item | Monthly |
|---|---|
| VM compute (730 hrs) | $330 |
| Additional costs | $19 |
| **Total** | **~$349/mo** |

**1-year Reserved:** ~$192/mo compute + $19 = **~$211/mo**

**Notes:** 8 vCPUs and 64 GB RAM gives comfortable headroom for 2 concurrent requests, OS overhead, and future growth. This is the tier to run in production.

---

### Option C — GPU (`Standard_NC4as_T4_v3`)

| Spec | Value |
|---|---|
| vCPUs | 4 |
| RAM | 28 GB |
| GPU | 1× NVIDIA Tesla T4 (16 GB VRAM) |
| On-demand price | $0.526/hr |

| Line item | Monthly |
|---|---|
| VM compute (730 hrs) | $384 |
| Additional costs | $19 |
| **Total** | **~$403/mo** |

**1-year Reserved:** ~$223/mo compute + $19 = **~$242/mo**

**Notes:** The NC4as_T4_v3 is Azure's entry-level GPU inference tier. The T4 has 16 GB VRAM which comfortably holds Donut (~500 MB weights + inference overhead). The Donut migration combined with the tiered pipeline reduces RAM requirements dramatically compared to v1.x (Florence-2-large-ft required ~7–8 GB total). A 4 GB machine is viable for low-traffic deployments. At $403/mo PAYG this is only ~$54/mo more than the recommended CPU tier — a reasonable premium for 5–10× faster inference.

> ⚠️ **GPU availability caveat:** NC-series VMs in East US can have limited availability. Have a fallback region (West US 2 or West Europe) configured.

---

## 4. Alternative Services

---

### AWS

**CPU Minimum equivalent: `r6i.xlarge`** (4 vCPU, 32 GB RAM, memory-optimised)

| Line item | Monthly |
|---|---|
| Compute on-demand ($0.252/hr × 730) | $184 |
| EBS gp3 volume 64 GB | ~$6 |
| Egress ~10 GB | ~$1 |
| Elastic IP | ~$4 |
| **Total** | **~$195/mo** |

**1-year Savings Plan:** ~$120/mo compute → **~$131/mo total**

**CPU Recommended equivalent: `m6i.2xlarge`** (8 vCPU, 32 GB RAM)  
Note: AWS general purpose gives 4 GB/vCPU vs Azure E-series 8 GB/vCPU, so to get 32 GB you need 8 vCPUs on `m6i`. For a true 64 GB equivalent use `r6i.2xlarge` (8 vCPU, 64 GB) at $0.504/hr → **~$390/mo PAYG.**

| Line item | Monthly (`m6i.2xlarge`) |
|---|---|
| Compute on-demand ($0.384/hr × 730) | $280 |
| EBS + IP + egress | ~$11 |
| **Total** | **~$291/mo** |

**GPU equivalent: `g4dn.xlarge`** (4 vCPU, 16 GB RAM, 1× NVIDIA T4 16 GB VRAM)

| Line item | Monthly |
|---|---|
| Compute on-demand ($0.526/hr × 730) | $384 |
| EBS + IP + egress | ~$11 |
| **Total** | **~$395/mo** |

**AWS vs Azure:** Broadly comparable pricing for equivalent specs. AWS has a slight edge on instance variety and reserved instance flexibility. Azure wins on RAM-per-vCPU in the E-series for this specific workload — you get 32 GB for 4 vCPUs vs AWS requiring 8 vCPUs (`m6i.2xlarge`) to reach the same RAM.

---

### Google Cloud Platform (GCP)

**CPU Minimum equivalent: `n2-highmem-4`** (4 vCPU, 32 GB RAM)

| Line item | Monthly |
|---|---|
| Compute on-demand (~$0.262/hr × 730) | $191 |
| Persistent disk 64 GB pd-ssd | ~$11 |
| Egress ~10 GB | ~$1 |
| Static IP | ~$3 |
| **Total** | **~$206/mo** |

**Committed Use Discount (1-year):** ~37% off compute → **~$135/mo total**

GCP also applies **Sustained Use Discounts (SUDs) automatically** — running a VM for the full month gives ~20% off the on-demand rate with no commitment required, bringing the effective PAYG rate to ~$153/mo total. This is a notable advantage over Azure and AWS which require explicit reservations for similar savings.

**CPU Recommended equivalent: `n2-standard-8`** (8 vCPU, 32 GB RAM)

| Line item | Monthly |
|---|---|
| Compute on-demand ($0.3885/hr × 730) | $284 |
| Disk + IP + egress | ~$15 |
| **Total** | **~$299/mo** |

With SUD (~20% auto-discount): **~$248/mo**

**Notes:** GCP's SUD is a genuine advantage for always-on workloads like this one. No reservation required — you just run it and the discount applies automatically at the end of the billing month.

---

### RunPod (GPU-focused cloud)

RunPod is a GPU marketplace primarily suited to ML inference and training workloads. It is not a general-purpose cloud — there is no managed networking, load balancing, or SLA, but for a stateless microservice reachable via a public endpoint it is viable.

**GPU equivalent: RTX 4000 Ada / A4000 pod** (16 GB VRAM, suitable for Florence-2)

| Line item | Monthly |
|---|---|
| Secure Cloud pod ~$0.40/hr × 730 hrs | $292 |
| Network volume 50 GB (persistent storage) | ~$7 |
| Egress (free on RunPod) | $0 |
| **Total** | **~$299/mo** |

For a T4-class GPU on Community Cloud (less reliable, shared infrastructure):

| Line item | Monthly |
|---|---|
| Community Cloud T4 ~$0.14–0.22/hr × 730 | $102–$161 |
| Network volume | ~$7 |
| **Total** | **~$109–$168/mo** |

**RunPod vs Azure GPU:** RunPod's Secure Cloud is ~$100/mo cheaper than Azure NC4as_T4_v3 for equivalent GPU performance. Community Cloud is up to $300/mo cheaper but carries interruption risk — unsuitable for a production always-on service. RunPod has zero egress fees, which Azure charges for.

**Disadvantages for this workload:** No SLA, no managed networking, no Azure ecosystem integration (relevant if your NestJS backend is already on Azure), manual container management. Best suited as a cost-optimised staging environment or for teams not already invested in a hyperscaler.

---

### Hetzner Cloud (budget European VPS)

Hetzner is a German provider with datacenters in Germany, Finland, and the US. Pricing is significantly cheaper than hyperscalers but with fewer managed services and no GPU instances currently available.

**CPU Minimum equivalent: `CCX23`** (4 dedicated vCPUs, 16 GB RAM)  
*(Note: Hetzner's shared CX series is not suitable — shared vCPUs will cause severe inference latency variance. Use CCX dedicated CPU series.)*

| Spec | CCX23 |
|---|---|
| vCPUs | 4 dedicated AMD EPYC |
| RAM | 16 GB |
| Storage | 160 GB NVMe SSD |
| Traffic | 20 TB included |
| Monthly (EUR) | ~€17.99 |
| **Monthly (USD approx.)** | **~$20** |

**CPU Recommended equivalent: `CCX33`** (8 dedicated vCPUs, 32 GB RAM)

| Spec | CCX33 |
|---|---|
| vCPUs | 8 dedicated AMD EPYC |
| RAM | 32 GB |
| Storage | 240 GB NVMe SSD |
| Monthly (EUR) | ~€34.99 |
| **Monthly (USD approx.)** | **~$38** |

> ⚠️ **Price adjustment notice:** Hetzner announced a price adjustment effective 1 April 2026. The figures above reflect post-adjustment pricing. Verify current EUR prices at hetzner.com/cloud and apply current EUR/USD conversion.

**Hetzner vs Azure:** 8–10× cheaper for CPU compute. The CCX33 at ~$38/mo vs Azure's ~$349/mo for comparable CPU/RAM is striking. The tradeoffs are real: no GPU option, European datacenters only (latency from Asia-Pacific is higher), no managed container registry, no SLA beyond basic uptime, and no ecosystem integration with Azure services.

**Best use case:** If your NestJS backend and this microservice can both move to Hetzner, the total cost savings are substantial. If you're keeping NestJS on Azure, Hetzner adds cross-provider egress complexity.

---

## 5. Recommendation

### Platform: Start on Azure, move to Hetzner if cost becomes a concern

If the rest of your stack (NestJS backend, Redis, etc.) is already on Azure, keep this service on Azure too. Cross-provider networking between NestJS on Azure and this service on Hetzner or RunPod adds latency, egress costs, and operational complexity that outweigh the savings at low volume.

**Recommended starting point: Azure `Standard_E4as_v5` (Option A, CPU minimum)**  
~$184/mo PAYG, ~$126/mo with 1-year reservation.

This is the pragmatic starting tier. It meets all requirements, leaves enough RAM headroom for low-to-medium load, and costs a fraction of the GPU tier. Run it here first and measure real inference times with real ID images before committing to a GPU upgrade.

---

### Is GPU worth it for this use case?

**Yes, if user-facing latency matters. No, if it is a background/async flow.**

The 5–10× inference speedup (20s → 3s) is significant if a user is waiting in a web UI for their ID to be verified. It is irrelevant if the verification runs asynchronously and the user is notified by email or push notification.

If synchronous: upgrade to `Standard_NC4as_T4_v3` at ~$403/mo PAYG (~$242/mo reserved). The user experience improvement justifies the ~$60/mo premium over the recommended CPU tier at reservation pricing.

If asynchronous: stay on CPU. The cost saving is substantial and the latency difference does not affect the user.

---

### Cost Optimisation Strategies

**1. Use Reserved Instances immediately if you commit to Azure.**  
1-year reserved pricing saves ~35–40% over PAYG. For an always-on service with no scale-to-zero, there is no reason to stay on PAYG beyond the initial evaluation period. At Option B recommended spec, reservation saves ~$138/mo (~$1,656/yr).

**2. Use a "Fat Container" strategy.**  
The provided Dockerfile already bakes in both Florence-2 and PaddleOCR weights. This eliminates the 30–60s cold-start download on every container restart, ensuring immediate availability and zero external network dependency at runtime.

**3. Use Azure Container Apps instead of a raw VM if you want managed infrastructure.**  
Azure Container Apps (consumption plan) charges per request rather than per hour, but the 30–60s model load time makes scale-to-zero unsuitable. Use the dedicated plan at a fixed monthly rate instead — it provides managed container runtime (no VM to patch) at similar cost to a VM.

**4. Consider Hetzner CCX33 for non-Azure deployments.**  
At ~$38/mo for 8 vCPU / 32 GB RAM, Hetzner is the most cost-effective CPU option if you are not locked into Azure. The savings ($38 vs $349/mo) fund significant other infrastructure. This is the right choice for a cost-sensitive early-stage product not yet committed to a hyperscaler.

**5. Monitor RAM, not CPU.**  
This service is memory-bound, not CPU-bound. If Azure Advisor recommends downsizing based on low CPU utilisation, ignore it — the RAM is doing the work, not the CPU. Set up a RAM utilisation alert at 80% rather than CPU.

---

## Summary Comparison Table

| Provider | Tier | vCPU | RAM | GPU | Monthly (PAYG) | Monthly (Reserved/Committed) |
|---|---|---|---|---|---|---|
| **Azure** | E4as_v5 (min CPU) | 4 | 32 GB | — | ~$184 | ~$126 |
| **Azure** | E8as_v5 (rec CPU) | 8 | 64 GB | — | ~$349 | ~$211 |
| **Azure** | NC4as_T4_v3 (GPU) | 4 | 28 GB | T4 16 GB | ~$403 | ~$242 |
| **AWS** | r6i.xlarge (min CPU) | 4 | 32 GB | — | ~$195 | ~$131 |
| **AWS** | g4dn.xlarge (GPU) | 4 | 16 GB | T4 16 GB | ~$395 | ~$250 est. |
| **GCP** | n2-highmem-4 (min CPU) | 4 | 32 GB | — | ~$206 | ~$135 (CUD) / ~$153 (SUD) |
| **GCP** | n2-standard-8 (rec CPU) | 8 | 32 GB | — | ~$299 | ~$248 (SUD auto) |
| **RunPod** | Secure Cloud A4000 (GPU) | varies | varies | RTX 4000 Ada | ~$299 | N/A |
| **Hetzner** | CCX23 (min CPU) | 4 | 16 GB | — | ~$20 | N/A |
| **Hetzner** | CCX33 (rec CPU) | 8 | 32 GB | — | ~$38 | N/A |

> **Flagged estimates:** Hetzner prices converted from EUR at approximate rate and subject to their April 2026 price adjustment. RunPod Community Cloud prices vary by availability. GCP SUD percentages are approximate. All hyperscaler reserved/committed rates are estimates based on published ~35–40% discount ranges — verify with each provider's pricing calculator before committing.

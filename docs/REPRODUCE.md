# DSv4 Ascend Serving — Reproduction Guide

End-to-end steps to clone, initialize containers, run four serving scenarios, and reproduce query3 smoke + GSM8K benchmarks on Ascend A3 (8×910 NPUs per node).

## Prerequisites

- Ascend driver + CANN installed on each host
- **8× Ascend 910 NPUs** per node (16 total for single-node dual-instance PD)
- Model weights: `DeepSeek-V4-Flash-w8a8-mtp` (70 `quant_model_weights-*.safetensors` shards)
- Docker with NPU device passthrough
- LAN connectivity between PD nodes (tested: prefill `192.168.0.223`, decode `192.168.0.110`, NIC `enp23s0f3`)
- Default container image: `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`

## 1. Clone & submodules

```bash
git clone --recursive https://YOUR_REMOTE/dsv4-ascend-serving.git
cd dsv4-ascend-serving
source versions.env
git submodule update --init --recursive
```

Pinned commits (see `versions.env`):

| Component | Commit |
|-----------|--------|
| LMCache | `a7934b79` (v0.4.5) |
| LMCache-Ascend | `7c0fb92` (`dsv4_support`) |
| kvcache-ops | `bab00e09` (multi-plane kernel) |

## 2. Per-node container init

Run on **each** Ascend host (adjust paths):

```bash
export REPO_ROOT=/path/to/dsv4-ascend-serving
export MODEL_MOUNT=/mnt/sdb/models          # parent dir of DeepSeek-V4-Flash-w8a8-mtp
export CACHE_MOUNT=$HOME/.cache
export NAME=vllm-ascend-dsv4-serving

bash scripts/container/run_container.sh
docker exec -u root "$NAME" bash /workspace/dsv4-serving/scripts/container/install_editable.sh
```

Expected install output ends with:

```
OK /workspace/LMCache/lmcache/__init__.py /workspace/LMCache-Ascend/lmcache_ascend/__init__.py
```

Container mount convention:

| Host | Container |
|------|-----------|
| `$REPO_ROOT` | `/workspace/dsv4-serving` |
| `$REPO_ROOT/LMCache` | `/workspace/LMCache` |
| `$REPO_ROOT/LMCache-Ascend` | `/workspace/LMCache-Ascend` |
| `$MODEL_MOUNT` | `/workspace/models` |

## 3. Four serving scenarios

All commands run **inside the container** unless noted.

### Scenario 1 — Single-node (L1 CPU only)

```bash
docker exec -it $NAME bash
cd /workspace/dsv4-serving
./scripts/serving/single_node.sh
```

- Port: **8008**
- LMCache: L1 CPU only (`kv_both`, no LocalDisk)

Stop: `Ctrl-C` or kill vLLM workers.

### Scenario 2 — Single-node + LocalDisk L2

```bash
./scripts/serving/single_node_disk.sh
```

- Port: **8008**
- L2: `file:///tmp/lmcache_disk` (500 GB default)

### Scenario 3 — 2-node PD (HCCL)

**Node A (prefiller + proxy):**

```bash
export LOCAL_IP=192.168.0.223
export PREFILL_HOST=192.168.0.223
export DECODE_HOST=192.168.0.110
export PROXY_HOST=192.168.0.223
export NIC_NAME=enp23s0f3
./scripts/serving/pd_prefiller_node.sh launch
```

**Node B (decoder):**

```bash
export LOCAL_IP=192.168.0.110
export PREFILL_HOST=192.168.0.223
export DECODE_HOST=192.168.0.110
export PROXY_HOST=192.168.0.223
export NIC_NAME=enp23s0f3
./scripts/serving/pd_decoder_node.sh launch
```

- Client port: **9100** (disagg proxy)
- Prefiller: 7100, Decoder: 7200

Stop on each node:

```bash
./scripts/serving/pd_prefiller_node.sh stop   # node A
./scripts/serving/pd_decoder_node.sh stop     # node B
```

### Scenario 4 — 2-node PD + LocalDisk L2

Same as scenario 3, but add `--disk` on both nodes:

```bash
./scripts/serving/pd_prefiller_node.sh --disk launch   # node A
./scripts/serving/pd_decoder_node.sh --disk launch     # node B
```

- L2 disk path per node: `/tmp/lmcache_disk_pd` (override via `LMCACHE_LOCAL_DISK`)
- Cross-node KV still uses HCCL PD; disk is per-node unless on shared FS

## 4. Smoke test (query3)

Long-context PD payload (~1024-token chunks). Run from a machine that can reach the proxy or single-node port.

**Single-node / disk:**

```bash
URL=http://127.0.0.1:8008/v1/chat/completions ./scripts/benchmark/query3.sh
```

**PD modes:**

```bash
URL=http://192.168.0.223:9100/v1/chat/completions ./scripts/benchmark/query3.sh
```

**Expected:**

- Response JSON contains `"finish_reason": "stop"`
- Prefiller log: `Stored 1024/1024` (or similar chunk store lines)
- Decoder log: `Retrieved 1024/1024`
- Output saved to `artifacts/smoke/smoke_response.json`

Collect LMCache evidence:

```bash
grep -E 'Stored|Retrieved|LocalDiskBackend|HCCL' logs/pd_disagg_dsv4_*/prefiller.log logs/pd_disagg_dsv4_*/decoder.log \
  > artifacts/lmcache_evidence.txt
```

## 5. GSM8K benchmark

Start serving first, then:

```bash
# Single-node
BASE_URL=http://127.0.0.1:8008/v1/chat/completions NUM_SAMPLES=200 ./scripts/benchmark/run_gsm8k.sh

# PD via proxy
BASE_URL=http://192.168.0.223:9100/v1/chat/completions NUM_SAMPLES=200 ./scripts/benchmark/run_gsm8k.sh
```

**Expected artifact:** `results/gsm8k_results.json` with `exact_match` metric (200 samples by default).

## 6. Throughput benchmark (vLLM bench)

Against a running single-node server:

```bash
HOST=127.0.0.1 PORT=8008 ./scripts/benchmark/run_bench.sh
```

Uses `vllm bench serve` with prefix repetition (30 prefixes × 30K tokens).

## 7. Artifact layout

Suggested structure after a full reproduction run:

```
artifacts/
├── smoke/
│   └── smoke_response.json
├── lmcache_evidence.txt
├── prefill/          # optional: copy of prefiller logs
└── decode/           # optional: copy of decoder logs
logs/
├── pd_disagg_dsv4_prefill_node/
│   ├── prefiller.log
│   ├── proxy.log
│   └── configs/
├── pd_disagg_dsv4_decode_node/
│   └── decoder.log
results/
└── gsm8k_results.json
```

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| Stale vLLM/proxy processes | `./scripts/serving/pd_*_node.sh stop` or `pkill -f vllm` |
| Server not ready in time | `export SERVER_WAIT_TIMEOUT=360` (default) |
| Multi-node decode uses wrong NPUs | Script auto-fallback: if `PREFILL_HOST != DECODE_HOST` and `DECODE_NPUS=8-15`, switches to `0-7` |
| Disk path permission denied | `mkdir -p /tmp/lmcache_disk_pd && chmod 1777 /tmp` |
| Model not found on decode node | Copy or NFS-mount model under `/workspace/models` |
| kvcache-ops build fails | Re-run `scripts/container/setup_kvcache_ops.sh` then `install_editable.sh` |
| Import errors after clone | Ensure submodules initialized and commits match `versions.env` |

## Environment reference

Key variables in `scripts/env/pd_dsv4_env.sh`:

- `PREFILL_HOST`, `DECODE_HOST`, `PROXY_HOST`, `DECODE_BIND_HOST`
- `NIC_NAME`, `LOCAL_IP` (HCCL/Gloo socket binding)
- `PREFILL_NPUS`, `DECODE_NPUS`, `TENSOR_PARALLEL_SIZE=8`
- `LMCACHE_CHUNK_SIZE=1024`, `PD_BUFFER_SIZE=1GiB`
- `SAVE_ONLY_FIRST_RANK=true` (MLA: TP0 stores/transfers KV)

For debug logging: append `--debug` to PD launchers.

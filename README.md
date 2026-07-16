# DSv4 Ascend Serving & Benchmark

Standalone reproduction repo for **DeepSeek-V4-Flash** on Ascend NPUs with **LMCache** / **LMCache-Ascend** (HCCL PD + optional LocalDisk L2).

## Quick start

```bash
git clone --recursive https://github.com/YOUR_ORG/dsv4-ascend-serving.git
cd dsv4-ascend-serving
source versions.env
git submodule update --init --recursive
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for container setup, four serving modes, query_example smoke, and GSM8K benchmarks.

## Serving modes

| Scenario | Script | Port |
|----------|--------|------|
| Single-node (L1 CPU) | `scripts/serving/single_node.sh` | 8008 |
| Single-node + disk | `scripts/serving/single_node_disk.sh` | 8008 |
| 2-node PD | `pd_prefiller_node.sh` + `pd_decoder_node.sh` | proxy 9100 |
| 2-node PD + disk | `pd_prefiller_node.sh --disk` + `pd_decoder_node.sh --disk` | proxy 9100 |

## Submodules

- [LMCache](https://github.com/LMCache/LMCache) @ `a7934b79` (v0.4.5)
- [LMCache-Ascend](https://github.com/marcobarlo/LMCache-Ascend) @ `7c0fb92` (`dsv4_support`)
- [kvcache-ops](https://atomgit.com/marco_barlo/kvcache-ops) @ pinned in `versions.env`

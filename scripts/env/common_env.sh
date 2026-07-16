#!/bin/bash
# Shared Ascend + LMCache environment for single-node serving.
# Source from serving scripts; do not execute directly.

export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_TRACK_USAGE="${LMCACHE_TRACK_USAGE:-false}"
export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-False}"
export LMCACHE_NUMA_MODE="${LMCACHE_NUMA_MODE:-auto}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
export LMCACHE_EXTRA_CONFIG="${LMCACHE_EXTRA_CONFIG:-{\"save_only_first_rank\": false}}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

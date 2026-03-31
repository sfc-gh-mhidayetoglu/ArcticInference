# demo_17 results

- **System**: 4x NVIDIA H200 (140 GiB each), 2 TiB CPU RAM, 28 TiB NVMe
- **Models**: 17 unquantized Qwen3 BF16 models
- **Errors**: None

## Summary

| Phase                  | Wall-clock |
|------------------------|------------|
| Register all 17 models | 131.0s     |
| Generate all 17 models | 40.6s      |

## Register times

| Model                        | Time    |
|------------------------------|---------|
| qwen3-4b-thinking-2507       | 34.5s   |
| qwen3-1.7b-base              | 43.8s   |
| qwen3-4b-instruct            | 44.1s   |
| qwen3-8b-instruct            | 46.7s   |
| qwen3-14b-base               | 55.4s   |
| qwen3-14b-instruct           | 61.4s   |
| qwen3-4b-instruct-2507       | 65.5s   |
| qwen3-8b-base                | 65.7s   |
| qwen3-0.6b-instruct          | 68.3s   |
| qwen3-1.7b-instruct          | 73.4s   |
| qwen3-30b-a3b-instruct-2507  | 75.9s   |
| qwen3-32b-instruct           | 84.5s   |
| qwen3-30b-a3b-thinking-2507  | 98.3s   |
| qwen3-0.6b-base              | 113.6s  |
| qwen3-30b-a3b-base           | 121.4s  |
| qwen3-4b-base                | 123.4s  |
| qwen3-30b-a3b-instruct       | 210.5s  |

Note: register time includes GPU wait time. With 4 GPUs and 17 models,
later models queue behind earlier ones.

## Generate times

| Model                        | Time    |
|------------------------------|---------|
| qwen3-4b-instruct            | 3.8s    |
| qwen3-8b-instruct            | 4.9s    |
| qwen3-14b-instruct           | 6.6s    |
| qwen3-1.7b-instruct          | 8.3s    |
| qwen3-1.7b-base              | 10.3s   |
| qwen3-32b-instruct           | 12.0s   |
| qwen3-4b-instruct-2507       | 16.9s   |
| qwen3-30b-a3b-instruct       | 18.7s   |
| qwen3-0.6b-base              | 21.3s   |
| qwen3-30b-a3b-instruct-2507  | 23.8s   |
| qwen3-30b-a3b-thinking-2507  | 23.8s   |
| qwen3-14b-base               | 26.2s   |
| qwen3-4b-thinking-2507       | 28.8s   |
| qwen3-8b-base                | 29.0s   |
| qwen3-4b-base                | 29.2s   |
| qwen3-0.6b-instruct          | 29.6s   |
| qwen3-30b-a3b-base           | 36.6s   |

Note: generate time includes restore + wake_up + h2d + scatter + inference +
sleep + checkpoint + GPU wait time. Parallelized across 4 GPUs.

## Instance primitive times (across all 17 models)

| Primitive        | Count | Min      | Avg      | Max      |
|------------------|-------|----------|----------|----------|
| init             | 17    | 20.226s  | 33.698s  | 58.490s  |
| attach           | 17    | 0.565s   | 8.942s   | 34.414s  |
| stage            | 17    | 0.267s   | 1.096s   | 1.655s   |
| sleep            | 34    | 0.042s   | 0.199s   | 0.757s   |
| checkpoint       | 34    | 1.149s   | 2.108s   | 4.112s   |
| restore          | 17    | 0.655s   | 2.361s   | 5.450s   |
| wake_up_weights  | 17    | 0.018s   | 0.405s   | 1.862s   |
| h2d              | 17    | 0.023s   | 0.628s   | 2.203s   |
| scatter          | 17    | 0.010s   | 0.210s   | 0.746s   |
| wake_up_kv_cache | 17    | 0.290s   | 0.633s   | 1.166s   |
| generate         | 17    | 0.090s   | 0.362s   | 1.222s   |

Note: sleep and checkpoint appear 34 times (once during register, once during
generate). init/attach/stage run only during register. The remaining primitives
run only during generate.

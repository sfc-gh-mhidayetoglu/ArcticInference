# Semi-Persistence Demo Results

## Registration (Cold Start)

| Model | Wait | Register | Total | Pinned |
|---|---|---|---|---|
| qwen3.5-0.8b | 155.9s | 24.0s | 180.0s | 1.59 GiB |
| qwen3.5-0.8b-base | 166.6s | 42.3s | 208.8s | 1.59 GiB |
| qwen3.5-2b | 127.4s | 28.8s | 156.2s | 4.12 GiB |
| qwen3.5-2b-base | 130.5s | 70.9s | 201.5s | 4.12 GiB |
| qwen3.5-4b | 99.9s | 31.0s | 130.9s | 8.46 GiB |
| qwen3.5-4b-base | 118.2s | 74.1s | 192.2s | 8.46 GiB |
| qwen3.5-9b | 65.2s | 34.9s | 100.1s | 17.53 GiB |
| qwen3.5-9b-base | 71.0s | 96.3s | 167.3s | 17.53 GiB |
| qwen3.5-35b-a3b-gptq-int4 | 179.7s | 57.5s | 237.2s | 20.92 GiB |
| qwen3.5-27b-gptq-int4 | 191.5s | 48.4s | 239.9s | 27.37 GiB |
| qwen3.5-27b-fp8 | 59.9s | 58.8s | 118.7s | 28.31 GiB |
| qwen3.5-35b-a3b-fp8 | 0.0s | 65.7s | 65.7s | 34.09 GiB |
| qwen3.5-27b | 0.0s | 60.0s | 60.0s | 50.96 GiB |
| qwen3.5-35b-a3b | 0.0s | 71.8s | 71.8s | 65.39 GiB |
| qwen3.5-35b-a3b-base | 0.0s | 128.3s | 128.3s | 65.39 GiB |

## Generate Round 1 (All Submitted at Once, 4 GPUs)

| Model | Wait | Restore | Generate | Checkpoint | Total |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 20.4s | 1.5s | 0.2s | 2.0s | 24.0s |
| qwen3.5-0.8b-base | 24.0s | 1.8s | 0.2s | 1.5s | 27.5s |
| qwen3.5-2b | 19.3s | 2.0s | 0.2s | 2.6s | 24.1s |
| qwen3.5-2b-base | 19.8s | 1.5s | 0.2s | 2.6s | 24.1s |
| qwen3.5-4b | 14.1s | 2.6s | 0.3s | 1.8s | 18.8s |
| qwen3.5-4b-base | 18.8s | 3.3s | 0.3s | 2.7s | 25.1s |
| qwen3.5-9b | 13.3s | 3.6s | 0.6s | 2.2s | 19.8s |
| qwen3.5-9b-base | 14.1s | 2.7s | 0.7s | 2.8s | 20.4s |
| qwen3.5-35b-a3b-gptq-int4 | 24.1s | 3.4s | 0.4s | 1.8s | 29.7s |
| qwen3.5-27b-gptq-int4 | 24.1s | 3.6s | 0.8s | 2.0s | 30.6s |
| qwen3.5-27b-fp8 | 11.7s | 4.0s | 1.1s | 2.5s | 19.3s |
| qwen3.5-35b-a3b-fp8 | 0.0s | 7.6s | 0.4s | 3.7s | 11.7s |
| qwen3.5-27b | 0.0s | 8.9s | 1.0s | 4.2s | 14.1s |
| qwen3.5-35b-a3b | 0.0s | 8.8s | 0.4s | 4.1s | 13.3s |
| qwen3.5-35b-a3b-base | 0.0s | 8.9s | 0.4s | 4.8s | 14.1s |

## Generate Round 2 (Sequential, 2s Sleep Between Submissions)

| Model | Wait | Restore | Generate | Checkpoint | Total |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 0.0s | 1.4s | 0.1s | 1.5s | 3.0s |
| qwen3.5-0.8b-base | 0.0s | 1.0s | 0.1s | 1.3s | 2.4s |
| qwen3.5-2b | 0.0s | 1.8s | 0.1s | 1.6s | 3.5s |
| qwen3.5-2b-base | 0.8s | 1.2s | 0.1s | 1.7s | 3.9s |
| qwen3.5-4b | 2.7s | 2.0s | 0.2s | 1.9s | 6.8s |
| qwen3.5-4b-base | 0.7s | 3.2s | 0.3s | 2.7s | 6.9s |
| qwen3.5-9b | 0.9s | 3.5s | 0.4s | 2.0s | 6.7s |
| qwen3.5-9b-base | 4.0s | 4.0s | 0.4s | 3.2s | 11.5s |
| qwen3.5-35b-a3b-gptq-int4 | 0.0s | 2.3s | 0.3s | 1.8s | 4.3s |
| qwen3.5-27b-gptq-int4 | 0.0s | 2.7s | 0.7s | 2.2s | 5.6s |
| qwen3.5-27b-fp8 | 0.6s | 3.7s | 1.0s | 2.7s | 8.0s |
| qwen3.5-35b-a3b-fp8 | 0.0s | 4.8s | 0.3s | 3.7s | 8.9s |
| qwen3.5-27b | 0.0s | 6.3s | 1.0s | 3.9s | 11.2s |
| qwen3.5-35b-a3b | 0.0s | 4.3s | 0.3s | 4.0s | 8.6s |
| qwen3.5-35b-a3b-base | 0.0s | 8.2s | 0.3s | 4.2s | 12.7s |

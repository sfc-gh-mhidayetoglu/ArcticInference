
from orchestrator import Orchestrator

vllm_config_1  = {"model": "Qwen/Qwen3.5-35B-A3B"}
vllm_config_2  = {"model": "Qwen/Qwen3.5-35B-A3B-FP8"}
vllm_config_3  = {"model": "Qwen/Qwen3.5-35B-A3B-Base"}
vllm_config_4  = {"model": "Qwen/Qwen3.5-27B"}
vllm_config_5  = {"model": "Qwen/Qwen3.5-27B-FP8"}
vllm_config_6  = {"model": "Qwen/Qwen3.5-9B"}
vllm_config_7  = {"model": "Qwen/Qwen3.5-9B-Base"}
vllm_config_8  = {"model": "Qwen/Qwen3.5-4B"}
vllm_config_9  = {"model": "Qwen/Qwen3.5-4B-Base"}
vllm_config_10 = {"model": "Qwen/Qwen3.5-2B"}
vllm_config_11 = {"model": "Qwen/Qwen3.5-2B-Base"}
vllm_config_12 = {"model": "Qwen/Qwen3.5-0.8B"}
vllm_config_13 = {"model": "Qwen/Qwen3.5-0.8B-Base"}
vllm_config_14 = {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"}
vllm_config_15 = {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4"}

if __name__ == "__main__":
    import datetime
    import os
    import sys
    log_path = f"demo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_file = open(log_path, "w")
    _log_fd = _log_file.fileno()
    os.dup2(_log_fd, 1)
    os.dup2(_log_fd, 2)
    sys.stdout = _log_file
    sys.stderr = _log_file

    Orchestrator.init(local_cache="/data-fast/model-cache")

    Orchestrator.print_status()

    Orchestrator.register("qwen3.5-35b-a3b",           vllm_config_1)
    Orchestrator.register("qwen3.5-35b-a3b-fp8",       vllm_config_2)
    Orchestrator.register("qwen3.5-35b-a3b-base",      vllm_config_3)
    Orchestrator.register("qwen3.5-27b",               vllm_config_4)
    Orchestrator.register("qwen3.5-27b-fp8",           vllm_config_5)
    Orchestrator.register("qwen3.5-9b",                vllm_config_6)
    Orchestrator.register("qwen3.5-9b-base",           vllm_config_7)
    Orchestrator.register("qwen3.5-4b",                vllm_config_8)
    Orchestrator.register("qwen3.5-4b-base",           vllm_config_9)
    Orchestrator.register("qwen3.5-2b",                vllm_config_10)
    Orchestrator.register("qwen3.5-2b-base",           vllm_config_11)
    Orchestrator.register("qwen3.5-0.8b",              vllm_config_12)
    Orchestrator.register("qwen3.5-0.8b-base",         vllm_config_13)
    Orchestrator.register("qwen3.5-35b-a3b-gptq-int4", vllm_config_14)
    Orchestrator.register("qwen3.5-27b-gptq-int4",     vllm_config_15)

    Orchestrator.wait_all()
    Orchestrator.print_status()

    prompt = "Explain the theory of relativity in one sentence."
    params = {"max_tokens": 64, "temperature": 0.7}

    Orchestrator.generate("qwen3.5-35b-a3b",           [prompt], params)
    Orchestrator.generate("qwen3.5-35b-a3b-fp8",       [prompt], params)
    Orchestrator.generate("qwen3.5-35b-a3b-base",      [prompt], params)
    Orchestrator.generate("qwen3.5-27b",               [prompt], params)
    Orchestrator.generate("qwen3.5-27b-fp8",           [prompt], params)
    Orchestrator.generate("qwen3.5-9b",                [prompt], params)
    Orchestrator.generate("qwen3.5-9b-base",           [prompt], params)
    Orchestrator.generate("qwen3.5-4b",                [prompt], params)
    Orchestrator.generate("qwen3.5-4b-base",           [prompt], params)
    Orchestrator.generate("qwen3.5-2b",                [prompt], params)
    Orchestrator.generate("qwen3.5-2b-base",           [prompt], params)
    Orchestrator.generate("qwen3.5-0.8b",              [prompt], params)
    Orchestrator.generate("qwen3.5-0.8b-base",         [prompt], params)
    Orchestrator.generate("qwen3.5-35b-a3b-gptq-int4", [prompt], params)
    Orchestrator.generate("qwen3.5-27b-gptq-int4",     [prompt], params)

    Orchestrator.wait_all()
    Orchestrator.print_status()

    from time import sleep
    Orchestrator.generate("qwen3.5-35b-a3b",           [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-35b-a3b-fp8",       [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-35b-a3b-base",      [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-27b",               [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-27b-fp8",           [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-9b",                [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-9b-base",           [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-4b",                [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-4b-base",           [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-2b",                [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-2b-base",           [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-0.8b",              [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-0.8b-base",         [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-35b-a3b-gptq-int4", [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3.5-27b-gptq-int4",     [prompt], params)

    Orchestrator.wait_all()
    Orchestrator.print_status()

    Orchestrator.remove_all()

    Orchestrator.wait_all()
    Orchestrator.print_status()


from orchestrator import Orchestrator

vllm_config_1  = {"model": "Qwen/Qwen3-32B"}
vllm_config_2  = {"model": "Qwen/Qwen3-14B"}
vllm_config_3  = {"model": "Qwen/Qwen3-8B"}
vllm_config_4  = {"model": "Qwen/Qwen3-4B"}
vllm_config_5  = {"model": "Qwen/Qwen3-1.7B"}
vllm_config_6  = {"model": "Qwen/Qwen3-0.6B"}
# vllm_config_7  = {"model": "Qwen/Qwen3-30B-A3B"}
# vllm_config_8  = {"model": "Qwen/Qwen3-30B-A3B-Base"}
# vllm_config_9  = {"model": "Qwen/Qwen3-14B-Base"}
# vllm_config_10 = {"model": "Qwen/Qwen3-8B-Base"}
# vllm_config_11 = {"model": "Qwen/Qwen3-4B-Base"}
# vllm_config_12 = {"model": "Qwen/Qwen3-1.7B-Base"}
# vllm_config_13 = {"model": "Qwen/Qwen3-0.6B-Base"}
# vllm_config_14 = {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507"}
# vllm_config_15 = {"model": "Qwen/Qwen3-4B-Instruct-2507"}
# vllm_config_16 = {"model": "Qwen/Qwen3-30B-A3B-Thinking-2507"}
# vllm_config_17 = {"model": "Qwen/Qwen3-4B-Thinking-2507"}

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

    Orchestrator.register("qwen3-32b-instruct",             vllm_config_1)
    Orchestrator.register("qwen3-14b-instruct",             vllm_config_2)
    Orchestrator.register("qwen3-8b-instruct",              vllm_config_3)
    Orchestrator.register("qwen3-4b-instruct",              vllm_config_4)
    Orchestrator.register("qwen3-1.7b-instruct",            vllm_config_5)
    Orchestrator.register("qwen3-0.6b-instruct",            vllm_config_6)
    # Orchestrator.register("qwen3-30b-a3b-instruct",         vllm_config_7)
    # Orchestrator.register("qwen3-30b-a3b-base",             vllm_config_8)
    # Orchestrator.register("qwen3-14b-base",                 vllm_config_9)
    # Orchestrator.register("qwen3-8b-base",                  vllm_config_10)
    # Orchestrator.register("qwen3-4b-base",                  vllm_config_11)
    # Orchestrator.register("qwen3-1.7b-base",                vllm_config_12)
    # Orchestrator.register("qwen3-0.6b-base",                vllm_config_13)
    # Orchestrator.register("qwen3-30b-a3b-instruct-2507",    vllm_config_14)
    # Orchestrator.register("qwen3-4b-instruct-2507",         vllm_config_15)
    # Orchestrator.register("qwen3-30b-a3b-thinking-2507",    vllm_config_16)
    # Orchestrator.register("qwen3-4b-thinking-2507",         vllm_config_17)

    Orchestrator.wait()
    Orchestrator.print_status()

    prompt = "Explain the theory of relativity in one sentence."
    params = {"max_tokens": 64, "temperature": 0.7}

    Orchestrator.generate("qwen3-32b-instruct",             [prompt], params)
    Orchestrator.generate("qwen3-14b-instruct",             [prompt], params)
    Orchestrator.generate("qwen3-8b-instruct",              [prompt], params)
    Orchestrator.generate("qwen3-4b-instruct",              [prompt], params)
    Orchestrator.generate("qwen3-1.7b-instruct",            [prompt], params)
    Orchestrator.generate("qwen3-0.6b-instruct",            [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-instruct",         [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-base",             [prompt], params)
    # Orchestrator.generate("qwen3-14b-base",                 [prompt], params)
    # Orchestrator.generate("qwen3-8b-base",                  [prompt], params)
    # Orchestrator.generate("qwen3-4b-base",                  [prompt], params)
    # Orchestrator.generate("qwen3-1.7b-base",                [prompt], params)
    # Orchestrator.generate("qwen3-0.6b-base",                [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-instruct-2507",    [prompt], params)
    # Orchestrator.generate("qwen3-4b-instruct-2507",         [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-thinking-2507",    [prompt], params)
    # Orchestrator.generate("qwen3-4b-thinking-2507",         [prompt], params)

    Orchestrator.wait()
    Orchestrator.print_status()

    from time import sleep
    Orchestrator.generate("qwen3-32b-instruct",             [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3-14b-instruct",             [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3-8b-instruct",              [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3-4b-instruct",              [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3-1.7b-instruct",            [prompt], params)
    sleep(2)
    Orchestrator.generate("qwen3-0.6b-instruct",            [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-instruct",         [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-base",             [prompt], params)
    # Orchestrator.generate("qwen3-14b-base",                 [prompt], params)
    # Orchestrator.generate("qwen3-8b-base",                  [prompt], params)
    # Orchestrator.generate("qwen3-4b-base",                  [prompt], params)
    # Orchestrator.generate("qwen3-1.7b-base",                [prompt], params)
    # Orchestrator.generate("qwen3-0.6b-base",                [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-instruct-2507",    [prompt], params)
    # Orchestrator.generate("qwen3-4b-instruct-2507",         [prompt], params)
    # Orchestrator.generate("qwen3-30b-a3b-thinking-2507",    [prompt], params)
    # Orchestrator.generate("qwen3-4b-thinking-2507",         [prompt], params)

    Orchestrator.wait()
    Orchestrator.print_status()

    Orchestrator.remove()

    Orchestrator.wait()
    Orchestrator.print_status()

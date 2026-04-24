
from orchestrator import Orchestrator as orch

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

    orch.init("/data-fast/image-cache/demo", [4, 5, 6, 7])

    orch.wait()
    orch.status()

    # orch.remove("model 1")
    # orch.remove("model 2")
    # orch.remove("model 3")
    # orch.remove("model 4")
    # orch.remove("model 5")
    # orch.remove("model 6")
    # orch.remove("model 7")
    # orch.remove("model 8")
    # orch.remove("model 9")
    # orch.remove("model 10")
    # orch.remove("model 11")
    # orch.remove("model 12")
    # orch.remove("model 13")
    # orch.remove("model 14")
    # orch.remove("model 15")

    orch.wait()
    orch.status()

    # init models
    # orch.register("model 1", {"model": "Qwen/Qwen3.5-35B-A3B"})
    # orch.register("model 2", {"model": "Qwen/Qwen3.5-35B-A3B-FP8"})
    # orch.register("model 3", {"model": "Qwen/Qwen3.5-35B-A3B-Base"})
    orch.register("model 4", {"model": "Qwen/Qwen3.5-27B"})
    orch.register("model 5", {"model": "Qwen/Qwen3.5-27B-FP8"})
    orch.register("model 6", {"model": "Qwen/Qwen3.5-9B"})
    orch.register("model 7", {"model": "Qwen/Qwen3.5-9B-Base"})
    orch.register("model 8", {"model": "Qwen/Qwen3.5-4B"})
    orch.register("model 9", {"model": "Qwen/Qwen3.5-4B-Base"})
    orch.register("model 10", {"model": "Qwen/Qwen3.5-2B"})
    orch.register("model 11", {"model": "Qwen/Qwen3.5-2B-Base"})
    orch.register("model 12", {"model": "Qwen/Qwen3.5-0.8B"})
    orch.register("model 13", {"model": "Qwen/Qwen3.5-0.8B-Base"})
    orch.register("model 14", {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"})
    orch.register("model 15", {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4"})

    orch.wait()
    orch.status()

    prompt = "Explain the theory of relativity in a paragraph"
    params = 1000

    # orch.generate("model 1", [prompt], params)
    # orch.generate("model 2", [prompt], params)
    # orch.generate("model 3", [prompt], params)
    orch.generate("model 4", [prompt], params)
    orch.generate("model 5", [prompt], params)
    orch.generate("model 6", [prompt], params)
    orch.generate("model 7", [prompt], params)
    orch.generate("model 8", [prompt], params)
    orch.generate("model 9", [prompt], params)
    orch.generate("model 10", [prompt], params)
    orch.generate("model 11", [prompt], params)
    orch.generate("model 12", [prompt], params)
    orch.generate("model 13", [prompt], params)
    orch.generate("model 14", [prompt], params)
    orch.generate("model 15", [prompt], params)

    orch.wait()
    orch.status()    

    # orch.generate("model 1", [prompt], params)
    # orch.generate("model 2", [prompt], params)
    # orch.generate("model 3", [prompt], params)
    orch.generate("model 4", [prompt], params)
    orch.generate("model 5", [prompt], params)
    orch.generate("model 6", [prompt], params)
    orch.generate("model 7", [prompt], params)
    orch.generate("model 8", [prompt], params)
    orch.generate("model 9", [prompt], params)
    orch.generate("model 10", [prompt], params)
    orch.generate("model 11", [prompt], params)
    orch.generate("model 12", [prompt], params)
    orch.generate("model 13", [prompt], params)
    orch.generate("model 14", [prompt], params)
    orch.generate("model 15", [prompt], params)

    orch.wait()
    orch.status()

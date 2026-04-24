import random
from orchestrator import Orchestrator as orch

PROMPTS = [
    "What is gravity?",
    "What is dark matter?",
    "What is an atom?",
    "Why is the sky blue?",
    "How do magnets work?",
    "What is photosynthesis?",
    "Explain TCP vs UDP.",
    "What is a neural network?",
    "What is DNA?",
    "How does GPS work?",
    "What is encryption?",
    "Write a haiku about rain.",
    "What causes tides?",
    "How do vaccines work?",
    "What is relativity?",
    "How does a CPU work?",
    "What is quantum entanglement?",
    "How do black holes form?",
    "What is the Doppler effect?",
    "Explain how WiFi works.",
]


def main() -> None:
    from time import sleep

    orch.init("/data-fast/image-cache/demo_7", [4, 5, 6])

    # orch.register("model 1", {"model": "Qwen/Qwen3.5-35B-A3B",           "gpu_memory_utilization": 0.8})
    orch.register("model 2", {"model": "Qwen/Qwen3.5-35B-A3B-FP8",       "gpu_memory_utilization": 0.8})
    # orch.register("model 3", {"model": "Qwen/Qwen3.5-35B-A3B-Base",      "gpu_memory_utilization": 0.8})
    orch.register("model 4",  {"model": "Qwen/Qwen3.5-27B",              "gpu_memory_utilization": 0.8})
    orch.register("model 5",  {"model": "Qwen/Qwen3.5-27B-FP8",          "gpu_memory_utilization": 0.8})
    orch.register("model 6",  {"model": "Qwen/Qwen3.5-9B",               "gpu_memory_utilization": 0.8})
    orch.register("model 7",  {"model": "Qwen/Qwen3.5-9B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 8",  {"model": "Qwen/Qwen3.5-4B",               "gpu_memory_utilization": 0.8})
    orch.register("model 9",  {"model": "Qwen/Qwen3.5-4B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 10", {"model": "Qwen/Qwen3.5-2B",               "gpu_memory_utilization": 0.8})
    orch.register("model 11", {"model": "Qwen/Qwen3.5-2B-Base",          "gpu_memory_utilization": 0.8})
    orch.register("model 12", {"model": "Qwen/Qwen3.5-0.8B",             "gpu_memory_utilization": 0.8})
    orch.register("model 13", {"model": "Qwen/Qwen3.5-0.8B-Base",        "gpu_memory_utilization": 0.8})
    orch.register("model 14", {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","gpu_memory_utilization": 0.8})
    orch.register("model 15", {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4",    "gpu_memory_utilization": 0.8})

    orch.wait()

    # orch.move("model 1", "sleep")
    # orch.move("model 2", "sleep")
    # orch.move("model 3", "sleep")
    # orch.move("model 4", "sleep")
    # orch.move("model 5", "sleep")
    # orch.move("model 6", "sleep")
    # orch.move("model 7", "sleep")
    # orch.move("model 8", "sleep")
    # orch.move("model 9", "sleep")
    # orch.move("model 10", "sleep")
    # orch.move("model 11", "sleep")
    # orch.move("model 12", "sleep")
    # orch.move("model 13", "sleep")
    # orch.move("model 14", "sleep")
    # orch.move("model 15", "sleep")
    # orch.wait()

    models = [
        # ("model 1",  172),
        ("model 2",  172),
        # ("model 3",  172),
        ("model 4",  64),
        ("model 5",  68),
        ("model 6",  179),
        ("model 7",  179),
        ("model 8",  270),
        ("model 9",  270),
        ("model 10", 476),
        ("model 11", 476),
        ("model 12", 625),
        ("model 13", 625),
        ("model 14", 208),
        ("model 15", 85),
    ]

    min_payload, max_payload = 5, 15
    interval = 2
    for i in range(1):
        for j in range(6):
            for name, tps in models:
                for _ in range(3):
                    max_tokens = tps * random.randint(min_payload, max_payload)
                    orch.generate(name, random.choice(PROMPTS), max_tokens)
                    sleep(interval)

        orch.wait()
        sleep(15)

    orch.move("model 1", "saved")
    orch.move("model 2", "saved")
    orch.move("model 3", "saved")
    orch.move("model 4", "saved")
    orch.move("model 5", "saved")
    orch.move("model 6", "saved")
    orch.move("model 7", "saved")
    orch.move("model 8", "saved")
    orch.move("model 9", "saved")
    orch.move("model 10", "saved")
    orch.move("model 11", "saved")
    orch.move("model 12", "saved")
    orch.move("model 13", "saved")
    orch.move("model 14", "saved")
    orch.move("model 15", "saved")
    orch.wait()

if __name__ == "__main__":
    main()

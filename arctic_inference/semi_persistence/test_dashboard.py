import argparse
import random
from time import sleep

from orch_client import RemoteOrchestrator as orch

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
    parser = argparse.ArgumentParser(
        description="Drive the orchestrator server through one workload phase.")
    parser.add_argument(
        "phase", type=int)
    args = parser.parse_args()

    orch.connect("http://localhost:8157")

    if args.phase == 1:
        orch.register("model 1", {"model": "Qwen/Qwen3.5-35B-A3B",           "gpu_memory_utilization": 0.8})
        orch.register("model 2", {"model": "Qwen/Qwen3.5-35B-A3B-FP8",       "gpu_memory_utilization": 0.8})
        orch.register("model 3", {"model": "Qwen/Qwen3.5-35B-A3B-Base",      "gpu_memory_utilization": 0.8})
        orch.register("model 4",  {"model": "Qwen/Qwen3.5-27B",              "gpu_memory_utilization": 0.8})
        orch.register("model 5",  {"model": "Qwen/Qwen3.5-27B-FP8",          "gpu_memory_utilization": 0.4})
        orch.register("model 6",  {"model": "Qwen/Qwen3.5-9B",               "gpu_memory_utilization": 0.4})
        orch.register("model 7",  {"model": "Qwen/Qwen3.5-9B-Base",          "gpu_memory_utilization": 0.4})
        orch.register("model 8",  {"model": "Qwen/Qwen3.5-4B",               "gpu_memory_utilization": 0.2})
        orch.register("model 9",  {"model": "Qwen/Qwen3.5-4B-Base",          "gpu_memory_utilization": 0.2})
        orch.register("model 10", {"model": "Qwen/Qwen3.5-2B",               "gpu_memory_utilization": 0.2})
        orch.register("model 11", {"model": "Qwen/Qwen3.5-2B-Base",          "gpu_memory_utilization": 0.2})
        orch.register("model 12", {"model": "Qwen/Qwen3.5-0.8B",             "gpu_memory_utilization": 0.2})
        orch.register("model 13", {"model": "Qwen/Qwen3.5-0.8B-Base",        "gpu_memory_utilization": 0.2})
        orch.register("model 14", {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","gpu_memory_utilization": 0.4})
        orch.register("model 15", {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4",    "gpu_memory_utilization": 0.4})
        orch.wait_all()

    if args.phase == 2:
        orch.move_all("sleep")
        orch.wait_all()

    if args.phase == 3:

        models = [
            ("model 1",  172),
            ("model 2",  172),
            ("model 3",  172),
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
        
        random.seed(0)
        min_payload, max_payload = 5, 15
        # interval = 0.1  # Poisson mean inter-arrival time (seconds)
        # num_requests = 26
        # for i in range(num_requests):
        #     name, tps = random.choice(models)
        #     max_tokens = tps * random.randint(min_payload, max_payload)
        #     orch.generate(name, random.choice(PROMPTS), max_tokens)
        #     sleep(random.expovariate(1.0 / interval))

        interval = 1
        for i in range(1):
            # for j in range(6):
            for j in range(3):
                for name, tps in models:
                    for _ in range(3):
                        max_tokens = tps * random.randint(min_payload, max_payload)
                        orch.generate(name, random.choice(PROMPTS), max_tokens)
                        sleep(interval)

        orch.wait_all()

    if args.phase == 4:
        orch.move_all("saved")
        orch.wait_all()

    if args.phase == 5:
        orch.move_all("checkpoint")
        orch.wait_all()

    if args.phase == 6:
        orch.generate_all("test", 1000)
        orch.wait_all()

    if args.phase == 7:
        orch.move("model 4", "checkpoint")
        orch.wait_all()

    if args.phase == 8:
        orch.generate("model 1", "test", 1000)
        orch.wait_all()

    if args.phase == 9:
        orch.move("model 1", "checkpoint")
        orch.wait_all()

    if args.phase == 10:
        orch.move_all("up")
        orch.wait_all()


if __name__ == "__main__":
    main()

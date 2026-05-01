import threading

from instance import Instance
from slots import Slots

configs = [
    {"model": "Qwen/Qwen3.5-35B-A3B-FP8",       "gpu_memory_utilization": 0.8},
    {"model": "Qwen/Qwen3.5-27B",              "gpu_memory_utilization": 0.8},
    {"model": "Qwen/Qwen3.5-27B-FP8",          "gpu_memory_utilization": 0.8},
    {"model": "Qwen/Qwen3.5-9B",               "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-9B-Base",          "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-4B",               "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-4B-Base",          "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-2B",               "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-2B-Base",          "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-0.8B",             "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-0.8B-Base",        "gpu_memory_utilization": 0.4},
    {"model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","gpu_memory_utilization": 0.8},
    {"model": "Qwen/Qwen3.5-27B-GPTQ-Int4",    "gpu_memory_utilization": 0.8},
]

images = [
    "/data-fast/image-cache/demo/model 1",
    "/data-fast/image-cache/demo/model 2",
    "/data-fast/image-cache/demo/model 3",
    "/data-fast/image-cache/demo/model 4",
    "/data-fast/image-cache/demo/model 5",
    "/data-fast/image-cache/demo/model 6",
    "/data-fast/image-cache/demo/model 7",
    "/data-fast/image-cache/demo/model 8",
    "/data-fast/image-cache/demo/model 9",
    "/data-fast/image-cache/demo/model 10",
    "/data-fast/image-cache/demo/model 11",
    "/data-fast/image-cache/demo/model 12",
    "/data-fast/image-cache/demo/model 13",
]

def main():

    Slots.init([2, 3])
    Slots.status()

    # inst.init(gpu=2)
    # inst.attach()
    # inst.repin()
    # inst.stage()
    # inst.unpin()
    # inst.sleep()
    # inst.checkpoint_cuda()
    # inst.save_image(IMAGE)
    # inst.wait()

    def run_one(i, config, image):
        inst = Instance(config)
        inst.load_image(image)

        level = 1 if config["gpu_memory_utilization"] > 0.5 else 2
        slot = Slots.allocate(level)
        Slots.status()

        inst.restore_cuda(gpu=slot.gpu_id)
        inst.wake_up_weights()
        inst.repin()
        inst.load_weights()
        inst.wake_up_kv_cache()
        inst.generate(["Hello, world!"], {})
        inst.sleep()
        inst.wait()

        Slots.deallocate(slot)
        Slots.status()

        inst.teardown()
        inst.wait()
        inst.remove()

    threads = [
        threading.Thread(target=run_one, args=(i, config, images[i]),
                         name=f"inst-{i}")
        for i, config in enumerate(configs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    Slots.remove()

if __name__ == "__main__":
    main()

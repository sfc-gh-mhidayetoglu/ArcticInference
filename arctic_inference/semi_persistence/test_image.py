import os, sys
from instance import Instance

MODEL = "Qwen/Qwen3-8B-FP8"
IMAGE_CACHE = "/data-fast/image-cache/test_image"

def main():
    inst = Instance({"model": MODEL, "enforce_eager": True})
    cache_hit = os.path.isfile(os.path.join(IMAGE_CACHE, "meta.json"))

    if cache_hit:
        print(f"[test] cache HIT — loading from {IMAGE_CACHE}")
        inst.load_image(IMAGE_CACHE).wait().print_status()
    else:
        print(f"[test] cache MISS — cold-starting on gpu=0, saving image")
        inst.init(gpu=2).wait().print_status()
        inst.attach().wait().print_status()
        inst.repin().wait().print_status()
        inst.stage().wait().print_status()
        inst.unpin().wait().print_status()
        inst.sleep().wait().print_status()
        inst.checkpoint_cuda().wait().print_status()
        inst.save_image(IMAGE_CACHE).wait().print_status()
        print(f"[test] image saved — continuing with same instance")

    inst.restore_cuda(gpu=3).wait().print_status()
    inst.wake_up_weights().wait().print_status()
    inst.repin().wait().print_status()
    inst.restore_weights().wait().print_status()
    inst.wake_up_kv_cache().wait().print_status()
    inst.generate(["Hello, world!"],{}).wait().print_status()
    inst.unpin().wait().print_status()
    inst.sleep().wait().print_status()
    inst.checkpoint_cuda().wait().print_status()

    inst.teardown().wait().remove()

if __name__ == "__main__":
    main()

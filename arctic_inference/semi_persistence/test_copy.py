from instance import Instance

MODEL = "Qwen/Qwen3-8B-FP8"
IMAGE_CACHE = "/data-fast/image-cache"

def main():

    inst = Instance({"model": MODEL, "enforce_eager": True})
    inst.init(gpu=0).wait().print_status()
    inst.attach().wait().print_status()
    inst.repin().wait().print_status()
    inst.stage().wait().print_status()
    inst.unpin().wait().print_status()
    inst.sleep().wait().print_status()
    inst.checkpoint().wait().print_status()
    inst.save(IMAGE_CACHE + "/test_copy").wait().print_status()

    # I can only restore on the same GPU as the checkpoint
    inst.load(IMAGE_CACHE + "/test_copy").wait().print_status()
    # inst.restore(gpu=0).wait().print_status()
    # inst.checkpoint().wait().print_status()

    # I can restore from any GPU
    inst.restore(gpu=2).wait().print_status()
    inst.wake_up_weights().wait().print_status()
    inst.repin().wait().print_status()
    inst.h2d().wait().print_status()
    inst.scatter().wait().print_status()
    inst.wake_up_kv_cache().wait().print_status()
    inst.generate(["Hello, world!"],{}).wait().print_status()

    inst.teardown().wait().remove()
    Instance.print_status()

if __name__ == "__main__":
    main()

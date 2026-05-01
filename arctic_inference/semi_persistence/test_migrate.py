from .instance import Instance


def main():

    vllm_config = {"model": "/data-fast/nvidia/Llama-3.1-70B-Instruct-FP8"}
    instance = Instance(vllm_config)

    Instance.print_status()

    instance.init(gpu=0)
    instance.attach()
    instance.wait()

    Instance.print_status()

    instance.sleep()
    instance.checkpoint_cuda()
    instance.wait()

    Instance.print_status()

    instance.restore_cuda(gpu=1)
    instance.wake_up_weights()
    instance.stage("/data-fast/nvidia/Llama-3.1-70B-Instruct-FP8")
    instance.load_weights()
    instance.wake_up_kv_cache()
    instance.wait()

    Instance.print_status()

    instance.teardown()
    instance.wait()

    Instance.print_status()

    instance.remove()

    Instance.print_status()

if __name__ == "__main__":
    main()

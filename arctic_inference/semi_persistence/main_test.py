from instance import Instance


def main():

    vllm_config_1 = {"model": "/data-fast/Qwen/Qwen3-1.7B"}
    vllm_config_2 = {"model": "/data-fast/nvidia/Llama-3.1-70B-Instruct-FP8"}
    vllm_config_3 = {"model": "/data-fast/Qwen/Qwen3-32B"}

    # -- Create and checkpoint instances 1 and 2 (parallel on different GPUs) --

    instance_1 = Instance(vllm_config_1)
    instance_2 = Instance(vllm_config_2)
    instance_3 = Instance(vllm_config_3)

    Instance.print_status()

    instance_1.init(gpu=0).attach().sleep()
    instance_2.init(gpu=1).attach().sleep()

    instance_1.wait()
    instance_2.wait()

    Instance.print_status()

    instance_1.checkpoint()
    instance_2.checkpoint()
    instance_1.wait()
    instance_2.wait()

    Instance.print_status()

    # -- Create instance 3 on GPU 0 after instance 1 finishes -----------------

    instance_3.after(instance_1).init(gpu=0).attach().sleep()

    instance_3.wait()
    instance_2.wait()

    Instance.print_status()

    instance_3.checkpoint()
    instance_3.wait()

    Instance.print_status()

    # All instances are checkpointed.
    #    GPU 0     GPU 1
    # | inst 1  | inst 2  |
    # | inst 3  |         |

    # -- Restore instance 1 and 2 ---------------------------------------------

    instance_1.restore(gpu=0)
    instance_1.wake_up(["weights"])
    instance_1.stage("/data-fast/Qwen/Qwen3-1.7B")
    instance_1.h2d()
    instance_1.scatter()
    instance_1.detach()
    instance_1.wake_up(["kv_cache"])

    instance_2.restore(gpu=1)
    instance_2.wake_up(["weights"])
    instance_2.stage("/data-fast/nvidia/Llama-3.1-70B-Instruct-FP8")
    instance_2.h2d()
    instance_2.scatter()
    instance_2.detach()
    instance_2.wake_up(["kv_cache"])

    instance_1.wait()
    instance_2.wait()

    #    GPU 0     GPU 1
    # | inst 1* | inst 2* |
    # | inst 3  |         |

    # -- Swap active model on GPU 0: hibernate 1, restore 3 -------------------

    instance_1.sleep().checkpoint()

    instance_3.after(instance_1)
    instance_3.restore(gpu=0)
    instance_3.wake_up(["weights"])
    instance_3.stage("/data-fast/Qwen/Qwen3-32B")
    instance_3.h2d()
    instance_3.scatter()
    instance_3.detach()
    instance_3.wake_up(["kv_cache"])
    instance_3.wait()

    #    GPU 0     GPU 1
    # | inst 1  | inst 2* |
    # | inst 3* |         |

    # -- Two small instances on GPU 1 -----------------------------------------

    vllm_config_4 = {"model": "/data-fast/Qwen/Qwen2.5-7B", "gpu_memory_utilization": 0.4}
    vllm_config_5 = {"model": "/data-fast/Qwen/Qwen3-1.7B", "gpu_memory_utilization": 0.4}

    instance_4 = Instance(vllm_config_4)
    instance_5 = Instance(vllm_config_5)

    instance_2.sleep().detach().checkpoint()

    instance_4.after(instance_2).init(gpu=1).attach()
    instance_5.after(instance_4).init(gpu=1).attach()
    instance_4.sleep().checkpoint()
    instance_5.sleep().checkpoint()
    instance_4.wait()
    instance_5.wait()

    # Restore instance 4 and 5 on the same GPU
    instance_4.restore(gpu=1).wake_up(["weights"]).stage("/data-fast/Qwen/Qwen2.5-7B").h2d().scatter().detach().wake_up(["kv_cache"])
    instance_5.restore(gpu=1).wake_up(["weights"]).stage("/data-fast/Qwen/Qwen3-1.7B").h2d().scatter().detach().wake_up(["kv_cache"])

    instance_4.wait()
    instance_5.wait()

    #        GPU 0     GPU 1
    # | inst 1  | inst 2  |
    # | inst 3* | inst 4* |
    #           | inst 5* |

    # -- Cleanup ---------------------------------------------------------------

    instance_1.teardown()
    instance_2.teardown()
    instance_3.teardown()
    instance_4.teardown()
    instance_5.teardown()

    instance_1.wait().remove()
    instance_2.wait().remove()
    instance_3.wait().remove()
    instance_4.wait().remove()
    instance_5.wait().remove()

    Instance.print_status()

if __name__ == "__main__":
    main()

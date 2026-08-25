## Rollout Replay: per-sequence output length for `n > 1`

This extends sampling so that, when `n > 1`, each of the `n` returned sequences
can be generated with its own output length. Combined with `ignore_eos`, each
sequence generates *exactly* the requested number of tokens.

The feature is part of the Arctic Inference patching system
(`arctic_inference/vllm/sampling.py`, `ParentRequestPatch`) and is applied
automatically by `apply_arctic_patches()` when Arctic Inference is enabled
(`ARCTIC_INFERENCE_ENABLED=1`, or when the plugin is loaded via
`vllm.plugins.load_general_plugins()`). There is nothing to install or patch.

Pass the per-child lengths through `SamplingParams.extra_args` under the
`max_tokens_n` key:

```python
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]

sampling_params = [SamplingParams(n=2,
                                 temperature=0.8,
                                 top_p=1.0,
                                 ignore_eos=True,
                                 extra_args={"max_tokens_n": [25, 50]},
                                ),
                   SamplingParams(n=3,
                                 temperature=0.8,
                                 top_p=1.0,
                                 ignore_eos=True,
                                 extra_args={"max_tokens_n": [5, 10, 15]},
                                ),
                   SamplingParams(n=1,
                                 temperature=0.8,
                                 top_p=1.0,
                                 max_tokens=100,
                                 ignore_eos=True,
                                 # extra_args={"max_tokens_n": [100]} would be
                                 # ineffective here since n = 1
                                ),
                   ]

outputs = llm.generate(prompts, sampling_params=sampling_params)
```

The number of resulting input and output tokens per sequence:
```
prompt 0 seq 0: input 5 output 25
prompt 0 seq 1: input 5 output 50
prompt 1 seq 0: input 7 output 5
prompt 1 seq 1: input 7 output 10
prompt 1 seq 2: input 7 output 15
prompt 2 seq 0: input 5 output 100
```

> Why `extra_args` instead of a `max_tokens_n=` kwarg? `SamplingParams` is a
> `msgspec.Struct` whose fields are fixed at class creation and which is
> serialized across the front-end → EngineCore process boundary. The plugin
> cannot add a real, serialized field at runtime, but `extra_args` is already
> such a field, so we ride on it.
>
> Requests that do not set `max_tokens_n` are unaffected (normal `max_tokens`
> behavior and child-param caching are preserved).

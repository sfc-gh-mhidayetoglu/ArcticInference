# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.parallel_sampling import ParentRequest

from arctic_inference.patching import ArcticPatch

logger = init_logger(__name__)

# Key under SamplingParams.extra_args that carries the per-child output lengths.
# We reuse the already-serialized ``extra_args`` dict instead of adding a new
# field because SamplingParams is a msgspec.Struct: its fields are fixed at
# class-creation and it is msgspec-encoded across the front-end -> EngineCore
# process boundary, so a monkeypatched attribute would neither be a real field
# nor survive serialization. ``extra_args`` is a genuine field, so it does.
MAX_TOKENS_N_KEY = "max_tokens_n"


class ParentRequestPatch(ArcticPatch[ParentRequest]):
    """Rollout-replay support: let ``n > 1`` requests give each child its own
    output length.

    Pass the lengths through ``SamplingParams.extra_args``::

        SamplingParams(n=2, ignore_eos=True,
                       extra_args={"max_tokens_n": [25, 50]})

    Child ``i`` is then generated with ``max_tokens = max_tokens_n[i]``.
    Requests that do not set ``max_tokens_n`` are unaffected (normal
    ``max_tokens`` behavior and child-param caching are preserved).
    """

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        seed = self.sampling_params.seed
        extra_args = self.sampling_params.extra_args or {}
        max_tokens_n = extra_args.get(MAX_TOKENS_N_KEY)
        # When max_tokens_n is set, every child needs a distinct max_tokens, so
        # the shared cache cannot be reused.
        if self.cached_child_sampling_params and max_tokens_n is None:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if max_tokens_n is not None:
            # Give each of the n children its own output length.
            child_sampling_params.max_tokens = max_tokens_n[index]
        if seed is None and max_tokens_n is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        elif seed is not None:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

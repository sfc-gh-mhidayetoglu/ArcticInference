"""Pytest fixtures shared across orchestrator tests.

After the migration completed in step 7 the orchestrator only has one
code path -- the explicit per-model pipeline (see ``pipeline.py`` and
``orchestrator_DESIGN.md``).  The historic ``ORCH_USE_PIPELINE`` flag
and its dual-mode parametrisation are gone; the ``pipeline_mode``
fixture is retained as a thin shim returning the constant
``"pipeline"`` so any pre-existing tests that depend on it (and any
new pipeline-mode regression tests) keep working without rewrites.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def pipeline_mode():
    """Compatibility shim: yields ``"pipeline"`` (the only mode now).

    Tests that branch on this value should treat any value other than
    ``"pipeline"`` as unreachable.  The legacy branch was removed in
    step 7 of the explicit-pipeline migration.
    """
    return "pipeline"

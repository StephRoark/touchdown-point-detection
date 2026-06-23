"""Shared pytest fixtures and Hypothesis configuration.

The design Testing Strategy specifies ``max_examples = 100`` and a
``deadline`` of 10000 ms. These are declared in ``pyproject.toml`` under
``[tool.hypothesis]`` for documentation, and registered here as a Hypothesis
profile so they are actually applied when the test suite runs.
"""

from hypothesis import settings

# Property-based tests run a minimum of 100 iterations each; the 10s deadline
# accommodates properties that exercise pipeline processing.
settings.register_profile("tdz", max_examples=100, deadline=10000)
settings.load_profile("tdz")

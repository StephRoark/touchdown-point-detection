"""Reproducibility & batch-level provenance (Task 23).

This module implements the two halves of Requirement 15:

Master-seed propagation (Req 15.2)
----------------------------------
:func:`derive_component_seeds` turns the single
``config.pipeline.master_random_seed`` into an independent integer sub-seed for
every stochastic component (LightGBM, the neural sequence model, the hybrid
residual boosters, data shuffling, ...). The derivation uses
:class:`numpy.random.SeedSequence` -- exactly the mechanism
:mod:`tdz.validation.splits` already uses for its split sub-seeds -- so a single
master seed deterministically fixes all random behaviour. The mapping is keyed
by component *name* (sorted before spawning), so a given ``(master_seed, name)``
pair always yields the same sub-seed regardless of how many other components are
requested or in what order.

Batch provenance (Req 15.3)
---------------------------
:class:`BatchProvenance` extends the per-record
:class:`tdz.assemble.Provenance` (reusing its ``config_hash`` /
``compute_config_hash`` and git-commit resolution rather than duplicating them)
with the batch-level fields the design mandates: the Python version, the
versions of the key numerical libraries, and the neural determinism mode that
was used (Req 15.1 / 15.2). One :class:`BatchProvenance` is resolved per output
batch via :func:`resolve_batch_provenance` and can be projected back down to a
per-record :class:`~tdz.assemble.Provenance` with
:meth:`BatchProvenance.to_record_provenance` for the assembler.

Reproducibility guarantees
--------------------------
Physics, change-point, LightGBM and geometry outputs are bit-identical across
two runs with the same seed / config / data (Req 15.1). The neural sequence
model is bit-identical when the explicit deterministic mode is enabled
(``config.pipeline.deterministic_mode``) and otherwise reproduces within a
documented tolerance; either way the mode used is recorded in the provenance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np

from tdz.assemble import Provenance, compute_config_hash, resolve_provenance
from tdz.config.schema import TDZConfig

__all__ = [
    "STOCHASTIC_COMPONENTS",
    "KEY_LIBRARIES",
    "derive_component_seeds",
    "derive_seed",
    "neural_deterministic_mode",
    "python_version",
    "library_versions",
    "compute_model_artifact_hash",
    "BatchProvenance",
    "resolve_batch_provenance",
]

#: Placeholder for a library that is not installed in the environment.
_NOT_INSTALLED: str = "not-installed"

#: The stochastic components whose randomness is fixed by the master seed
#: (Req 15.2). Every learned/stochastic estimator derives its own seed from the
#: master seed via :func:`derive_component_seeds` under one of these names.
STOCHASTIC_COMPONENTS: tuple[str, ...] = (
    "lightgbm",
    "sequence_model",
    "hybrid_residual",
    "data_shuffle",
    "validation_splits",
)

#: Key numerical libraries whose versions are recorded with every batch
#: (Req 15.3). Distribution names (as known to the installed-package metadata),
#: not import names -- ``scikit-learn`` rather than ``sklearn``.
KEY_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "torch",
    "pyproj",
    "ruptures",
    "filterpy",
)


# ---------------------------------------------------------------------------
# Master-seed propagation (Req 15.2)
# ---------------------------------------------------------------------------


def _name_key(name: str) -> int:
    """A stable (non-salted) integer key for a component name.

    Python's builtin ``hash`` is per-process salted, so we use a SHA-256 digest
    to key the seed derivation reproducibly across runs and processes.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big")


def derive_component_seeds(
    master_seed: int, component_names: Iterable[str] = STOCHASTIC_COMPONENTS
) -> dict[str, int]:
    """Derive an independent integer sub-seed per component from ``master_seed``.

    Uses :class:`numpy.random.SeedSequence` (mirroring
    :func:`tdz.validation.splits.make_validation_splits`) so a single master
    seed fixes all random behaviour while giving each component a *statistically
    independent* stream. Each sub-seed is derived from the pair
    ``(master_seed, sha256(name))``, so a given ``(master_seed, name)`` pair
    always maps to the same sub-seed regardless of the set/order of other
    requested components.

    Returns
    -------
    dict[str, int]
        ``{component_name -> seed}`` with each seed a 32-bit unsigned integer
        (the range LightGBM ``random_state`` / ``master_random_seed`` accept).
    """
    names = sorted(set(component_names))
    seeds: dict[str, int] = {}
    for name in names:
        seq = np.random.SeedSequence([int(master_seed), _name_key(name)])
        seeds[name] = int(seq.generate_state(1, dtype=np.uint32)[0])
    return seeds


def derive_seed(master_seed: int, component_name: str) -> int:
    """Derive the single sub-seed for one named component (see :func:`derive_component_seeds`)."""
    return derive_component_seeds(master_seed, (component_name,))[component_name]


def neural_deterministic_mode(config: TDZConfig) -> bool:
    """The neural deterministic-execution mode selected in config (Req 15.2)."""
    return bool(getattr(config.pipeline, "deterministic_mode", True))


# ---------------------------------------------------------------------------
# Environment capture (Req 15.3)
# ---------------------------------------------------------------------------


def python_version() -> str:
    """The running Python version (e.g. ``"3.12.4"``)."""
    return platform.python_version()


def library_versions(
    libraries: Iterable[str] = KEY_LIBRARIES,
) -> dict[str, str]:
    """Resolve installed versions of the key numerical libraries (Req 15.3).

    A library that is not installed is recorded as ``"not-installed"`` rather
    than omitted, so the provenance record has a stable, complete key set.
    """
    versions: dict[str, str] = {}
    for name in libraries:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = _NOT_INSTALLED
    return versions


def compute_model_artifact_hash(path: str) -> str:
    """Return a stable SHA-256 hex digest of a model artifact file (Req 15.3)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Batch-level provenance (Req 15.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchProvenance:
    """Reproducibility provenance recorded once per output batch (Req 15.3).

    Extends the per-record :class:`tdz.assemble.Provenance` (``data_version`` /
    ``code_commit`` / ``config_hash`` / ``model_artifact_hash``) with the
    batch-level environment fields the design mandates: the Python version, the
    key numerical-library versions, and the neural determinism mode that was
    used (Req 15.1 / 15.2).
    """

    data_version: str
    code_commit: str
    config_hash: str
    python_version: str
    library_versions: Mapping[str, str]
    neural_deterministic_mode: bool
    model_artifact_hash: Optional[str] = None

    def to_record_provenance(self) -> Provenance:
        """Project down to the per-record :class:`~tdz.assemble.Provenance`.

        The assembler stamps every :class:`~tdz.models.TouchdownResult` with the
        per-record subset; the batch-level fields live on the batch envelope.
        """
        return Provenance(
            data_version=self.data_version,
            code_commit=self.code_commit,
            config_hash=self.config_hash,
            model_artifact_hash=self.model_artifact_hash,
        )

    def to_dict(self) -> dict:
        """Plain-dict view (for serialising the batch envelope / logging)."""
        return {
            "data_version": self.data_version,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "python_version": self.python_version,
            "library_versions": dict(self.library_versions),
            "neural_deterministic_mode": self.neural_deterministic_mode,
            "model_artifact_hash": self.model_artifact_hash,
        }


def resolve_batch_provenance(
    config: TDZConfig,
    *,
    data_version: str = "unknown",
    code_commit: Optional[str] = None,
    model_artifact_hash: Optional[str] = None,
    neural_deterministic_mode_used: Optional[bool] = None,
    libraries: Iterable[str] = KEY_LIBRARIES,
) -> BatchProvenance:
    """Build the :class:`BatchProvenance` for a run (Req 15.3).

    Reuses :func:`tdz.assemble.resolve_provenance` for the ``config_hash`` and
    the local git commit (``code_commit`` may be supplied to stay hermetic in
    tests) and augments it with the Python version, the key-library versions,
    and the neural determinism mode. When ``neural_deterministic_mode_used`` is
    not given it is taken from ``config.pipeline.deterministic_mode``, so the
    provenance records the mode that was actually configured (Req 15.2).
    """
    base = resolve_provenance(
        config,
        data_version=data_version,
        code_commit=code_commit,
        model_artifact_hash=model_artifact_hash,
    )
    mode = (
        neural_deterministic_mode(config)
        if neural_deterministic_mode_used is None
        else bool(neural_deterministic_mode_used)
    )
    return BatchProvenance(
        data_version=base.data_version,
        code_commit=base.code_commit,
        config_hash=base.config_hash,
        model_artifact_hash=base.model_artifact_hash,
        python_version=python_version(),
        library_versions=library_versions(libraries),
        neural_deterministic_mode=mode,
    )

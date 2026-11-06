"""Results must not depend on PYTHONHASHSEED (us_hank._determinism).

Each case runs ``_hash_probe.py`` in a fresh interpreter, because string-hash
salting is fixed per process: a demo-grid GE Jacobian and a spending IRF are
hashed bit for bit.
"""
import os
import subprocess
import sys
from pathlib import Path

from sequence_jacobian.classes.jacobian_dict import JacobianDict

import us_hank  # noqa: F401  (installs the patch)
from us_hank import _determinism

PROBE = Path(__file__).with_name('_hash_probe.py')
SEEDS = ('0', '1', '2', '3')


def _hash(mode, seed):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, str(PROBE), mode], env=env,
                         capture_output=True, text=True, check=True, timeout=300)
    return out.stdout.strip().splitlines()[-1]


def test_importing_us_hank_installs_the_sorted_methods():
    assert JacobianDict.compose is _determinism._compose
    assert JacobianDict.apply is _determinism._apply


def test_jacobian_and_irf_are_identical_across_hash_seeds():
    assert len({_hash('patched', s) for s in SEEDS}) == 1


def test_the_toolkit_alone_does_drift_across_hash_seeds():
    """Without the patch the same computation differs between seeds (at the
    1e-15 level), which is what the patch is for. If this ever fails, the
    toolkit no longer iterates sets there and the patch can go."""
    assert len({_hash('unpatched', s) for s in SEEDS}) > 1

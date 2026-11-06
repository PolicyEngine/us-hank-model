"""Make the toolkit's Jacobian algebra independent of ``PYTHONHASHSEED``.

``sequence_jacobian`` 1.0.0 (pinned in pyproject.toml) sums intermediate
Jacobian products in the iteration order of a plain ``set`` of variable
names in two places:

* ``JacobianDict.compose`` (classes/jacobian_dict.py:165):
  ``m_list = tuple(set(self.inputs) & set(J.outputs))``
* ``JacobianDict.apply`` (classes/jacobian_dict.py:191):
  ``inputs = x.keys() & set(self.inputs)``

String hashing is salted per process, so that order, and with it the
floating-point summation order, changes from one process to the next and the
general-equilibrium Jacobian and IRFs drift at about 1e-15 between runs
(seen in CI runs 42146220866, 42146453172, 42146710366). The 4-decimal pins
never noticed; any bit-level comparison across processes would.

This module replaces those two methods with copies that iterate the same
names in sorted order. Nothing else changes: same products, same sums, a
fixed order. ``ORIGINAL`` keeps the toolkit's methods so a test can show the
patch is what removes the drift.
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from sequence_jacobian.classes.impulse_dict import ImpulseDict
from sequence_jacobian.classes.jacobian_dict import JacobianDict

PATCHED_VERSION = '1.0.0'

ORIGINAL = {'compose': JacobianDict.compose, 'apply': JacobianDict.apply}


def _compose(self, J):
    """Returns self @ J (toolkit 1.0.0 body, intermediates in sorted order)."""
    if self.T is not None and J.T is not None and self.T != J.T:
        raise ValueError(f'Trying to multiply JacobianDicts with inconsistent '
                         f'dimensions {self.T} and {J.T}')
    o_list = self.outputs
    m_list = tuple(sorted(set(self.inputs) & set(J.outputs)))
    i_list = J.inputs
    J_om = self.nesteddict
    J_mi = J.nesteddict
    J_oi = {}
    for o in o_list:
        J_oi[o] = {}
        for i in i_list:
            Jout = None
            for m in m_list:
                if m in J_om[o] and i in J_mi[m]:
                    if Jout is None:
                        Jout = J_om[o][m] @ J_mi[m][i]
                    else:
                        Jout += J_om[o][m] @ J_mi[m][i]
            if Jout is not None:
                J_oi[o][i] = Jout
    return JacobianDict(J_oi, o_list, i_list)


def _apply(self, x):
    """Returns J @ x (toolkit 1.0.0 body, inputs in sorted order)."""
    x = ImpulseDict(x)
    inputs = sorted(x.keys() & set(self.inputs))
    J_oi = self.nesteddict
    y = {}
    for o in self.outputs:
        y[o] = np.zeros(x.T)
        J_i = J_oi[o]
        for i in inputs:
            if i in J_i:
                y[o] += J_i[i] @ x[i]
    return x | ImpulseDict(y, T=x.T)


def install():
    """Install the sorted-order methods (idempotent). Refuses to patch a
    toolkit version whose method bodies it has not been checked against."""
    found = version('sequence-jacobian')
    if found != PATCHED_VERSION:
        raise ImportError(
            f'us_hank._determinism was written against sequence-jacobian '
            f'{PATCHED_VERSION}, found {found}; '
            're-check jacobian_dict.py compose/apply before bumping the pin')
    JacobianDict.compose = _compose
    JacobianDict.apply = _apply


def uninstall():
    JacobianDict.compose = ORIGINAL['compose']
    JacobianDict.apply = ORIGINAL['apply']


install()

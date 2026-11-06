"""Helper for test_determinism: prints a hash of a demo-grid GE Jacobian and
IRF. ``python _hash_probe.py patched|unpatched``."""
import hashlib
import sys

import numpy as np

from us_hank import _determinism, model

if sys.argv[1] == 'unpatched':
    _determinism.uninstall()
cal = dict(model.PRODUCTION_CALIBRATION, nB=10, nA=16, nK=4)
ss_m, m = model.build_models()
cali = ss_m.solve_steady_state(cal, unknowns={'beta': 0.976, 'chi1': 6.5},
                               targets={'asset_mkt': 0., 'B': 'Bh'},
                               solver='broyden_custom')
ss = m.steady_state(cali)
G = m.solve_jacobian(ss, model.UNKNOWNS, model.TARGETS, model.EXOGENOUS, T=60)
h = hashlib.sha256()
for o in ('Y', 'C', 'I', 'pi', 'r', 'tax'):
    for i in model.EXOGENOUS:
        h.update(np.ascontiguousarray(G[o][i]).tobytes())
irf = model.shock('fiscal_spending', 0.01, 0.8, T=60, ss=ss, G_jac=G)
h.update(np.ascontiguousarray(irf['Y']).tobytes())
print(h.hexdigest())

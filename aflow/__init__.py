"""fsai-aflow: AFlow workflow optimization package.

Provides the ``aflow.*`` namespace so external consumers can write::

    from aflow.scripts.optimizer import Optimizer

Internally, ``aflow.X`` is an alias for the top-level ``X`` package.
A meta-path finder ensures both import paths resolve to the *same*
module objects, avoiding isinstance / identity mismatches.
"""

import importlib
import sys

_SUBPACKAGES = ("scripts", "benchmarks", "data", "log_viz")


class _AflowAliasImporter:
    """Redirect ``aflow.X.Y`` imports to ``X.Y``."""

    def find_module(self, fullname, _path=None):
        if fullname == "aflow" or not fullname.startswith("aflow."):
            return None
        rest = fullname[len("aflow.") :]
        top = rest.split(".")[0]
        if top in _SUBPACKAGES:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real_name = fullname[len("aflow.") :]
        real_mod = importlib.import_module(real_name)
        sys.modules[fullname] = real_mod
        return real_mod


sys.meta_path.insert(0, _AflowAliasImporter())

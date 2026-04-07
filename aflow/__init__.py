"""fsai-aflow: AFlow workflow optimization package.

Provides the ``aflow.*`` namespace so external consumers can write::

    from aflow.scripts.optimizer import Optimizer

Internally, ``aflow.X`` is an alias for the top-level ``X`` package.
A meta-path finder ensures both import paths resolve to the *same*
module objects, avoiding isinstance / identity mismatches.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys

_SUBPACKAGES = ("scripts", "benchmarks", "data", "log_viz")


class _AflowAliasImporter(importlib.abc.MetaPathFinder):
    """Redirect ``aflow.X.Y`` imports to ``X.Y``.

    Uses find_spec (PEP 451) so it works on Python 3.12+ where the
    legacy find_module / load_module protocol was removed.
    """

    def find_spec(self, fullname, _path, _target=None):
        if fullname == "aflow" or not fullname.startswith("aflow."):
            return None
        rest = fullname[len("aflow.") :]
        top = rest.split(".")[0]
        if top not in _SUBPACKAGES:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AflowAliasLoader())


class _AflowAliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        real_name = spec.name[len("aflow.") :]
        real_mod = importlib.import_module(real_name)
        sys.modules[spec.name] = real_mod
        return real_mod

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _AflowAliasImporter())

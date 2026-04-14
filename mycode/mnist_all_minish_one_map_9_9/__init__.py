"""Expose the repository root under the legacy import path.

The original code imported modules from
`mycode.mnist_all_minish_one_map_9_9`. In this repository snapshot the
modules live at the repository root, so this package extends its search path
to include that location.
"""

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]

__path__ = [str(_PKG_DIR), str(_REPO_ROOT)]

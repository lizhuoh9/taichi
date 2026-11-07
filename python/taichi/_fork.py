# Added in the fork lizhuoh9/taichi on 2026-11-06 (see FORK.md): the fork's marker and its launch fast path switch.
"""The marker of the Taichi fork lizhuoh9/taichi, re-exported as ``taichi._fork``.

A program that needs this fork reads the marker before ``ti.init``: stock Taichi has no ``taichi._fork``.

NAME: "lizhuoh9/taichi", the repository of the fork.
VERSION: the version of the fork, also the version of its wheel: Taichi's release, then a local label that counts the
    fork's rounds of changes (FORK.md).  A local label satisfies ``taichi==1.7.4`` (PEP 440), and no public index carries
    one, so it never collides with an official release.  The C++ core is the release's, so ``ti.__version__`` stays
    (1, 7, 4).
BASE: the upstream tag the fork's branch starts from.
LAUNCH_FASTPATH: whether the launch fast paths of taichi.lang.kernel_impl are on in this process.  The environment
    variable TI_LAUNCH_FASTPATH is read once, at import: unset or "1" turns them on, "0" turns them off (for A/B timing
    within one installation), and any other value raises ValueError, so a typo never runs the other way silently.
"""

import os

NAME = "lizhuoh9/taichi"
VERSION = "1.7.4+3"
BASE = "v1.7.4"
LAUNCH_FASTPATH_VARIABLE = "TI_LAUNCH_FASTPATH"


def _read_launch_fastpath(environ):
    """LAUNCH_FASTPATH from the environment mapping (module docstring)."""
    value = environ.get(LAUNCH_FASTPATH_VARIABLE)
    if value is None or value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{LAUNCH_FASTPATH_VARIABLE} must be 1 (on, as when it is unset) or 0 (off); got {value!r}")


LAUNCH_FASTPATH = _read_launch_fastpath(os.environ)

__all__ = []

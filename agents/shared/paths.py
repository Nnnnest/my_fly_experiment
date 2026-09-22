"""Single source of truth for where the fly-brain library's data files
(mb_circuit.npz, mb_groups*.npz, etc.) live. FlyBrainAPI's own default
(path=".") is resolved against the process's CWD at runtime, not against
fly_api.py's location -- breaks whenever a script is invoked from
anywhere other than the fly-brain root itself. This file sits at
<fly-brain-root>/my_experiments/agents/shared/paths.py, so three levels
up from here is always <fly-brain-root>, regardless of the caller's CWD.
"""
import os

FLY_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

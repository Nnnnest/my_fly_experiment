"""FlyBrainAPI(mode='mb') instances reused within a process, reset to pristine state between runs.
slot=0 is the default; slot=1 is a second independent brain (self-play)."""
import numpy as np
from core.paths import FLY_ROOT, live_alpn

_SLOTS = {}


class _Adapter:
    def __init__(self, patterns):
        self.patterns = patterns
        self.n_actions = 1


def fresh_brain(slot=0):
    s = _SLOTS.get(slot)
    if s is None:
        from fly_api import FlyBrainAPI
        b = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        b.enable_scaling()
        if 0 in _SLOTS and getattr(_SLOTS[0]["brain"], "_mbc_shared", None) is not None:
            b._mbc_shared = _SLOTS[0]["brain"]._mbc_shared      # frozen ALPN->KC matrix: share it
        _SLOTS[slot] = {"brain": b, "wM0": b.wM0.copy(), "ref_mb": b.ref_mb.copy(), "caches": {}}
        return b
    b = s["brain"]
    b.wM[:] = s["wM0"]
    b.ref_mb[:] = s["ref_mb"]
    b._km_tag = set()
    b._hprev = np.zeros(b.N, dtype=np.float32)
    b._kc_fam = None
    b.asleep, b.sleep_S = False, 0.0
    for c in s["caches"].values():
        c.refresh()
    return b


def cache_for(name, patterns, slot=0):
    from agents.shared.mb_value_cache import MBValueCache
    s = _SLOTS[slot]
    c = s["caches"].get(name)
    if c is None:
        c = MBValueCache(s["brain"], _Adapter(patterns))
        res = c.verify(n=min(8, len(patterns)))
        if not res["ok"]:
            raise RuntimeError(f"value cache disagrees with brain.step: {res}")
        s["caches"][name] = c
    c.refresh()
    return c


def value_fn(name="generic", slot=0):
    rng = np.random.default_rng(0)
    live = live_alpn()
    pats = np.zeros((4, len(_SLOTS[slot]["brain"].ALPN)), np.float32)
    for i in range(4):
        pats[i, rng.choice(live, 60, replace=False)] = rng.uniform(1, 5, 60)
    c = cache_for(name, pats, slot)
    return c, (lambda vecs: c._codes(np.atleast_2d(vecs)) @ c.Wc)

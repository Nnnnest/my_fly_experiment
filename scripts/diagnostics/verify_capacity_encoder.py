"""CapacityDigitPairEncoder: equals MultiDigitPairEncoder at the original group size; fixed, disjoint groups otherwise.
    python -m scripts.diagnostics.verify_capacity_encoder"""
import numpy as np

from core import paths
from encoders.comparison.multi_digit_pair_encoder import MultiDigitPairEncoder
from encoders.comparison.capacity_digit_pair_encoder import CapacityDigitPairEncoder

live = paths.live_alpn()
for d in (2, 3, 4, 5, 9):
    ref = MultiDigitPairEncoder(live, n_digits=d, seed=3)
    same = CapacityDigitPairEncoder(live, n_digits=d, per_position=ref.group_size, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(ref.groups_a + ref.groups_b, same.groups_a + same.groups_b))
    assert np.array_equal(ref.encode(12345 % 10 ** d, 54321 % 10 ** d), same.encode(12345 % 10 ** d, 54321 % 10 ** d))
    g16 = CapacityDigitPairEncoder(live, n_digits=d, per_position=16, seed=3)
    allg = np.concatenate(g16.groups_a + g16.groups_b)
    assert len(allg) == len(set(allg.tolist())) == 2 * d * 16
    print(f"digits {d}: original group size {ref.group_size}, matches; g16 groups disjoint, total {len(allg)} ALPNs")
print("OK")

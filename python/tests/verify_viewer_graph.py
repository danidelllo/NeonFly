"""Full-data validation (not part of the small unit suite)."""
import numpy as np
import pandas as pd

from python.viewer.prepare import DATA, acquire


def main():
    load = lambda n: np.load(DATA / (n + ".npy"), mmap_mode="r")
    ids = load("ids")
    np.testing.assert_array_equal(ids, np.sort(pd.read_csv(acquire("neurons")).root_id))
    rng = np.random.default_rng(783)
    sample = rng.choice(ids, 32, replace=False)
    expected = {int(i): {} for i in sample}
    total = 0
    for chunk in pd.read_csv(acquire("connections_no_threshold"), chunksize=400000,
                             usecols=["pre_root_id", "post_root_id", "syn_count"]):
        total += int(chunk.syn_count.sum())
        for a, b, n in chunk.loc[chunk.pre_root_id.isin(sample)].itertuples(index=False, name=None):
            expected[a][b] = expected[a].get(b, 0) + n
    ip, ix, iw = (load("in_"+k) for k in ("indptr", "indices", "data"))
    op, ox, ow = (load("out_"+k) for k in ("indptr", "indices", "data"))
    assert int(ip[-1]) == int(op[-1]) == int(load("bundles")[:, 2].sum(dtype=np.int64))
    assert total == int(iw.sum(dtype=np.int64)) == int(ow.sum(dtype=np.int64))
    for a, neighbors in expected.items():
        i = np.searchsorted(ids, a)
        sl = slice(op[i], op[i+1])
        assert dict(zip(ids[ox[sl]], ow[sl])) == neighbors
        for b, n in neighbors.items():
            j = np.searchsorted(ids, b)
            k = ip[j] + np.searchsorted(ix[ip[j]:ip[j+1]], i)
            assert k < ip[j+1] and ix[k] == i and iw[k] == n
    print(f"{len(ids)} root IDs, {ip[-1]} directed pairs, {total} synapses; "
          f"both indices matched the raw table for {sum(map(len, expected.values()))} sampled pairs.")


if __name__ == "__main__":
    main()

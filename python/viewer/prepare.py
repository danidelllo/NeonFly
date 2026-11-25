"""Acquire pinned Codex tables and build full incoming/outgoing adjacency indices."""
import base64
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

DATA = Path(__file__).resolve().parents[2] / ".deps/flywire783"
FILES = {
    "neurons": ("1733428446832038", "9gMz6eQSQWC5sgOxcSpvkQ=="),
    "classification": ("1771945205700934", "cBwPrwVM7dtGC+pOh9amJA=="),
    "coordinates": ("1733428431583877", "96gSD4MiQF/svuwu4dIG5w=="),
    "connections_no_threshold": ("1733428417714928", "gzLA8DrNJBn1b5u8ACoVVA=="),
}


def acquire(name):
    gen, md5 = FILES[name]
    dest = DATA / "raw" / (name + ".csv.gz")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"https://storage.googleapis.com/flywire-data/codex/data/fafb/783/{dest.name}?generation={gen}",
                     timeout=120) as src, dest.with_suffix(".part").open("wb") as out:
            shutil.copyfileobj(src, out)
        dest.with_suffix(".part").replace(dest)
    with dest.open("rb") as f:
        if hashlib.file_digest(f, "md5").digest() != base64.b64decode(md5):
            raise ValueError(f"FlyWire digest mismatch: {dest}")
    return dest


def positions3d(ids):
    coords = pd.read_csv(acquire("coordinates")).drop_duplicates("root_id").set_index("root_id").loc[ids]
    xyz = np.array([s[1:-1].split() for s in coords.position], np.float32)
    xyz -= (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    return xyz * (0.925 / np.abs(xyz).max())


def prepare():
    paths = {n: acquire(n) for n in FILES}
    nodes = pd.read_csv(paths["neurons"]).set_index("root_id").sort_index()
    ids = nodes.index
    if len(ids) != 139255 or not ids.is_unique:
        raise ValueError("unexpected neuron roster")
    cls = pd.read_csv(paths["classification"]).set_index("root_id").loc[ids].fillna("")
    coords = pd.read_csv(paths["coordinates"]).drop_duplicates("root_id").set_index("root_id").loc[ids]
    xyz = np.array([s[1:-1].split() for s in coords.position], np.float32)
    xy = xyz[:, :2].copy()
    xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
    xy *= 1.85 / np.abs(xy).max() / 2
    pre, post, counts = [], [], []
    for frame in pd.read_csv(paths["connections_no_threshold"], chunksize=400000,
                             usecols=["pre_root_id", "post_root_id", "syn_count"]):
        a, b = ids.get_indexer(frame.pre_root_id), ids.get_indexer(frame.post_root_id)
        if a.min() < 0 or b.min() < 0 or frame.syn_count.min() < 1:
            raise ValueError("edge not in the roster or invalid count; refusing to discard it")
        pre.append(a.astype(np.int32))
        post.append(b.astype(np.int32))
        counts.append(frame.syn_count.to_numpy(np.int32))
    incoming = coo_matrix((np.concatenate(counts), (np.concatenate(post), np.concatenate(pre))),
                          shape=(len(ids), len(ids))).tocsr()
    synapses = sum(int(c.sum()) for c in counts)
    if incoming.data.sum() != synapses:
        raise ValueError("synapse count mismatch")
    del pre, post, counts
    outgoing = incoming.T.tocsr()
    def save(name, data):
        np.save(DATA / name, data, allow_pickle=False)
    for name, graph in (("in", incoming), ("out", outgoing)):
        for field in ("indptr", "indices", "data"):
            save(name + "_" + field, getattr(graph, field))
    save("ids", ids.to_numpy(np.int64))
    save("xy", xy)
    save("sensory", np.flatnonzero(cls["class"] == "visual").astype(np.int32))
    save("output", np.flatnonzero(cls.super_class.isin(["descending", "motor"])).astype(np.int32))
    save("sign", np.where(nodes.nt_type == "GABA", -1, 1).astype(np.float32))
    labels = (cls.super_class + " / " + cls["class"] + " / " + cls.side).tolist()
    (DATA / "labels.json").write_text(json.dumps(labels))
    # Every directed pair contributes to an overview line between spatial bins.
    grid = np.clip(((xy + 1) * 16).astype(int), 0, 31)
    bins = grid[:, 0] + 32 * grid[:, 1]
    edges = np.zeros(1024 * 1024, np.int64)
    for i in range(0, len(ids), 2048):
        hi = min(len(ids), i+2048)
        origins = np.repeat(bins[i:hi], np.diff(outgoing.indptr[i:hi+1]))
        targets = bins[outgoing.indices[outgoing.indptr[i]:outgoing.indptr[hi]]]
        np.add.at(edges, origins*1024+targets, 1)
    idx = np.flatnonzero(edges)
    if int(edges.sum()) != incoming.nnz:
        raise ValueError("overview lost edges")
    save("bundles", np.column_stack((idx//1024, idx%1024, edges[idx])).astype(np.int32))
    meta = dict(nodes=len(ids), edges=incoming.nnz, synapses=synapses, release="FAFB v783",
                visual=int((cls["class"] == "visual").sum()), sources=FILES)
    (DATA / "manifest.json").write_text(json.dumps(meta))
    print(meta)


if __name__ == "__main__":
    prepare()

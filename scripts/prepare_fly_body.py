"""Bake the pinned Apache-2.0 NeuroMechFly neutral pose into a viewer mesh.

Raw source assets stay in .deps. The derived mesh retains upstream licensing.
No connectome, training feature, or game state is changed.
"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "152506d3471646f009480c81f34aefbaec29a6e5"
BASE = f"https://raw.githubusercontent.com/NeLy-EPFL/fly-svg-maker/{COMMIT}/"
CACHE = ROOT / ".deps/fly-body"
OUT = ROOT / "python/viewer/web/assets/fly"


def acquire(name):
    # Versioned cache: record the revision separately from exploratory downloads.
    file = CACHE / COMMIT / name
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(BASE + name, timeout=30) as r:
            file.write_bytes(r.read())
    return file.read_bytes()


def bake():
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads(acquire("assets/model.json"))
    def rest(name):
        cfg = model["rest"][name]
        w, x, y, z = cfg["quat"]
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        m[:3, 3] = cfg["pos"]
        return m
    transforms = {model["root"]: rest(model["root"])}
    for parent, child in model["joints"]:
        joint = np.eye(4)
        for dof in model["dofs"]:
            if (dof["parent"], dof["child"]) != (parent, child):
                continue
            axis = model["axisVector"][dof["axis"]]
            angle = np.deg2rad(model["neutralDeg"].get(dof["name"], 0))
            joint[:3, :3] = joint[:3, :3] @ Rotation.from_rotvec(np.array(axis)*angle).as_matrix()
        transforms[child] = transforms[parent] @ rest(child) @ joint
    error = max(np.max(np.abs(transforms[s][:3, 3]-model["reference"]["neutralJointPositions"][s]))
                for s in model["segments"])
    if error > 1e-6:
        raise ValueError(f"pose reconstruction differs from upstream: {error}")

    triangles, parts, sources = [], [], {}
    stl_type = np.dtype([("normal", "<f4", 3), ("vertex", "<f4", (3, 3)), ("attribute", "<u2")])
    for name in model["segments"]:
        cfg = model["meshes"][name]
        path = "assets/meshes/" + cfg["file"]
        raw = acquire(path)
        sources[path] = hashlib.sha256(raw).hexdigest()
        count = int.from_bytes(raw[80:84], "little")
        if len(raw) != 84 + count*50:
            raise ValueError(f"invalid binary STL: {path}")
        p = np.frombuffer(raw, dtype=stl_type, offset=84)["vertex"].astype(np.float64)*model["meshScale"]
        if cfg["mirror"]:
            p[:, :, 1] *= -1
            p = p[:, [0, 2, 1]]
        transform = transforms[name]
        p = p @ transform[:3, :3].T + transform[:3, 3]
        triangles.append(p)
        parts.append(dict(name=name, first=sum(len(t)*3 for t in triangles[:-1]), count=len(p)*3))

    head = np.concatenate([t for t, part in zip(triangles, parts) if part["name"] in ("c_head", "l_eye", "r_eye")])
    lo, hi = head.reshape(-1, 3).min(0), head.reshape(-1, 3).max(0)
    anchor = (lo+hi)/2
    anchor[0] -= .05*(hi[0]-lo[0])
    anchor[2] += .10*(hi[2]-lo[2])
    scale = 1.85 / (.72*(hi[1]-lo[1]))
    # Body AP/LR/DV -> connectome display LR/down/AP. Illustrative registration.
    p = (np.concatenate(triangles)-anchor)*scale
    p = p[:, :, [1, 2, 0]] * [1, -1, 1]
    normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    packed = np.concatenate((p, np.repeat(normals[:, None], 3, axis=1)), axis=2).astype("<f4")
    packed.tofile(OUT / "body.bin")
    manifest = dict(source="NeuroMechFly v2 via NeLy-EPFL/fly-svg-maker", commit=COMMIT,
                    url="https://github.com/NeLy-EPFL/fly-svg-maker", license="Apache-2.0",
                    registration="Illustrative alignment between different specimens; not an anatomical registration",
                    pose="upstream neutral pose", fk_error=float(error), vertices=len(p)*3,
                    bounds=[p.reshape(-1, 3).min(0).tolist(), p.reshape(-1, 3).max(0).tolist()],
                    parts=parts, sources=sources, sha256=hashlib.sha256((OUT/"body.bin").read_bytes()).hexdigest())
    (OUT / "model.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "NOTICE").write_bytes(acquire("assets/NOTICE"))
    with urlopen("https://raw.githubusercontent.com/NeLy-EPFL/flygym/main/LICENSE", timeout=30) as r:
        (OUT / "LICENSE").write_bytes(r.read())
    print(json.dumps({k: manifest[k] for k in ("vertices", "bounds", "fk_error", "sha256")}))


if __name__ == "__main__":
    bake()

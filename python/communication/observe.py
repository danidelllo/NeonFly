"""Read one real NEONSKY end-of-frame observation from the local mod."""
import argparse
import json
import os
import socket
import struct
from pathlib import Path


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("bridge disconnected mid-message")
    return data


def receive(stream, capture):
    magic, metadata_len, image_len = struct.unpack("<4sII", read_exact(stream, 12))
    if magic != b"NFL1" or not 0 < metadata_len <= 65536 or image_len > 32 << 20:
        raise ValueError("invalid bridge response header")
    meta = json.loads(read_exact(stream, metadata_len))
    if "sensorValid" in meta:
        meta["rays"] = dict(valid=meta["sensorValid"], insideView=meta["sensorInsideView"], range=meta["sensorRange"],
            distance=meta["sensorDistance"], visible=meta["sensorVisible"], kind=meta["sensorKind"],
            visibleStart=meta.get("sensorVisibleStart", [0.]*32),
            geometryDistance=meta["sensorGeometryDistance"], geometryKind=meta["sensorGeometryKind"],
            vx=meta["sensorVx"], vy=meta["sensorVy"], screenX=meta["sensorScreenX"], screenY=meta["sensorScreenY"])
    if meta.get("ok") is not True:
        raise RuntimeError(meta.get("error", "unknown bridge error"))
    image = read_exact(stream, image_len)
    if capture:
        if (len(image) < 24 or image[:16] != b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"
                or struct.unpack(">II", image[16:24]) != (meta["width"], meta["height"])):
            raise ValueError("missing or inconsistent PNG")
    elif image:
        raise ValueError("unexpected status payload")
    return meta, image


def request(op, direction=0, ticks=1, expected=None):
    payload = b"NFL1" + bytes([op])
    if op in (3, 4, 5, 6, 7, 8, 9, 10):
        key = (Path(os.environ["LOCALAPPDATA"]) / "NeonFly" / "session.key").read_bytes()
        if len(key) != 32:
            raise ValueError("invalid local session credential")
        payload += key
        if op in (5, 7, 8, 9):
            if direction not in (range(36) if op >= 8 else (-1, 0, 1)) or not 1 <= ticks <= 32:
                raise ValueError("invalid ship action")
            payload += bytes([direction if op >= 8 else direction + 1, ticks])
            if op >= 7:
                if expected is None:
                    raise ValueError("guarded step requires an episode identity")
                payload += struct.pack("<iii", *expected)
    with socket.create_connection(("127.0.0.1", 47651), timeout=10) as sock:
        sock.sendall(payload)
        with sock.makefile("rb") as stream:
            return receive(stream, op in (2, 5, 6, 7, 8, 9, 10))


def observe(capture):
    return request(2 if capture else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, help="capture PNG instead of status only")
    args = parser.parse_args()
    metadata, frame = observe(args.png is not None)
    if args.png:
        args.png.write_bytes(frame)
    print(json.dumps(metadata))

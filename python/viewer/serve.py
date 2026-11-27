"""Loopback graph inspector with same-origin action-boundary training controls."""
import argparse
import json
import multiprocessing as mp
import queue
import re
import struct
import threading
import time
import sys
import secrets
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

import numpy as np

from python.viewer.prepare import DATA, positions3d
from python.viewer.policy import actor_packet

WEB = Path(__file__).with_name("web")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47652)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-run", help="Run PPO instead of the untrained readout")
    parser.add_argument("--warm-start", type=Path, help="Initialize a new run from an existing actor")
    parser.add_argument("--replay-best", help="Play three episodes with this run's frozen best policy")
    parser.add_argument("--resume", action="store_true", help="Continue --train-run from its latest checkpoint")
    parser.add_argument("--resume-from", type=Path, help="Resume --train-run from a specific full checkpoint")
    parser.add_argument("--continuous", action="store_true", help="Train until stopped instead of stopping at the configured update limit")
    parser.add_argument("--config", default="configs/reservoir_ppo.toml")
    parser.add_argument("--env-config", default="configs/ship.toml")
    parser.add_argument("--observe", action="store_true", help="Display the live service on 47652 without controlling the game")
    args = parser.parse_args()
    if args.resume_from:
        if args.resume or not args.train_run:
            parser.error("--resume-from requires --train-run and cannot accompany --resume")
        args.resume = args.resume_from
    if args.observe and (args.port == 47652 or args.train_run or args.replay_best or args.resume or args.warm_start):
        parser.error("--observe needs a separate port and cannot start a controller")
    if args.resume and not args.train_run:
        parser.error("--resume requires --train-run")
    if args.continuous and not args.train_run:
        parser.error("--continuous requires --train-run")
    if args.replay_best and args.train_run or args.warm_start and (args.resume or not args.train_run):
        parser.error("choose one of fresh training, warm start, resume, or replay")
    active_run = args.train_run or args.replay_best
    encoder = tomllib.loads(Path(args.config).read_text()).get('encoder') if args.train_run else None
    algorithm = tomllib.loads(Path(args.config).read_text()).get('algorithm') if args.train_run else None
    if args.replay_best:
        import torch
        encoder = torch.load(Path('checkpoints')/args.replay_best/'best.pt',map_location='cpu',weights_only=True)['signature']['config'].get('encoder')
        algorithm = torch.load(Path('checkpoints')/args.replay_best/'best.pt',map_location='cpu',weights_only=True)['signature']['config'].get('algorithm')
    from python.vision.range_sensor import RADAR_ENCODER
    from python.vision.radar_readout import readout_neurons
    active_outputs = (np.load(DATA/'output.npy').astype('<i4') if algorithm in ('brain_awr','brain_ppo')
                      else readout_neurons(encoder).astype('<i4'))
    if active_run and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", active_run):
        parser.error("invalid run name")
    load = lambda n: np.load(DATA / (n + ".npy"), mmap_mode="r")
    meta = (DATA/"manifest.json").read_bytes()
    ids, xy, bundles = (load(k) for k in ("ids", "xy", "bundles"))
    labels = json.loads((DATA/"labels.json").read_text())
    graph = {s: tuple(load(s+"_"+k) for k in ("indptr", "indices", "data")) for s in ("in", "out")}
    static = {"/meta": ("application/json", meta), "/xy": ("application/octet-stream", xy.tobytes()),
              "/bundles": ("application/octet-stream", bundles.tobytes())}
    xyz = positions3d(ids)
    bins = np.clip(((xy + 1) * 16).astype(int), 0, 31)
    bins = bins[:, 0] + 32*bins[:, 1]
    sizes = np.maximum(1, np.bincount(bins, minlength=1024))
    centroids = np.column_stack([np.bincount(bins, weights=axis, minlength=1024)/sizes for axis in xyz.T])
    for route, array in (("/xyz", xyz), ("/centroids", centroids.astype("<f4")),
                         ("/inputs", load("sensory")), ("/outputs", active_outputs)):
        static[route] = ("application/octet-stream", array.tobytes())
    from python.vision.connectome import sensory_mapping
    visual, slots, resource_neurons = sensory_mapping(load("sensory"))
    if encoder == RADAR_ENCODER:
        visual, slots = visual[:0],slots[:0]
        static['/inputs'] = ('application/octet-stream',resource_neurons.astype('<i4').tobytes())
    from python.vision.range_sensor import sensor_mapping
    range_neurons, range_slots = sensor_mapping()
    static["/encoding"] = ("application/json", json.dumps(dict(side=42,
        visual=visual.tolist(), slots=slots.tolist(), resources=resource_neurons.tolist(),
        rangeNeurons=range_neurons.tolist(), rangeSlots=range_slots.tolist())).encode())
    static["/range-neurons"] = ("application/octet-stream", range_neurons.astype("<i4").tobytes())
    for url, name, mime in (("/", "index.html", "text/html"), ("/ui.js", "ui.js", "text/javascript"),
                            ("/ui.css", "ui.css", "text/css"), ("/wiring.js", "wiring.js", "text/javascript"),
                            ("/fly-body.js", "fly-body.js", "text/javascript"),
                            ("/input-maps.js", "input-maps.js", "text/javascript"),
                            ("/radar.js", "radar.js", "text/javascript")):
        static[url] = (mime, (WEB/name).read_bytes())
    for name,mime in (("body.bin", "application/octet-stream"), ("model.json", "application/json"),
                      ("NOTICE", "text/plain"), ("LICENSE", "text/plain")):
        static["/assets/fly/"+name] = (mime, (WEB/"assets/fly"/name).read_bytes())
    frames = mp.Queue(maxsize=2)
    stop = mp.Event()
    from python.viewer.control import TrainingControl
    from python.environment.ship import ShipConfig
    control = TrainingControl() if args.train_run and ShipConfig.load(args.env_config).controlled else None
    control_token = secrets.token_urlsafe(32)
    latest = None
    error = None
    last_visit = time.monotonic()
    policy_frame = None
    def note_frame(data):
        nonlocal policy_frame
        size, = struct.unpack_from("<I", data)
        m = json.loads(data[4:4+size])
        policy_frame = (m.get("run", ""), "train" if m.get("training") else "replay" if m.get("replay") else "demo", m.get("update"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, mime, body):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        def do_GET(self):
            nonlocal last_visit
            if self.headers.get("Host") != f"127.0.0.1:{args.port}":
                self.send_error(403)
                return
            u = urlsplit(self.path)
            if args.observe and u.path in ('/encoding','/inputs','/outputs','/range-neurons'):
                try:
                    with urlopen('http://127.0.0.1:47652'+u.path,timeout=5) as upstream:
                        self.send(upstream.status,upstream.headers.get('Content-Type'),upstream.read())
                except (URLError,TimeoutError):
                    self.send(503,'text/plain',b'Training mapping unavailable')
            elif u.path in static:
                self.send(200, *static[u.path])
            elif u.path == "/actor":
                key = policy_frame
                try:
                    if key is None:
                        raise ValueError("Waiting for policy frame")
                    data = actor_packet(*key, width=len(load("output")))
                except (ValueError, OSError) as exc:
                    self.send(503, "text/plain", str(exc).encode())
                    return
                header = json.dumps(key).encode()
                self.send(200, "application/octet-stream", struct.pack("<I", len(header))+header+data)
            elif args.observe and u.path in ("/progress", "/heartbeat", "/frame", "/control"):
                last_visit = time.monotonic()
                try:
                    with urlopen("http://127.0.0.1:47652"+u.path, timeout=4) as upstream:
                        data = upstream.read()
                        if u.path == "/frame" and upstream.status == 200:
                            note_frame(data)
                        self.send(upstream.status, upstream.headers.get_content_type(), data)
                except HTTPError as exc:
                    self.send(exc.code, "text/plain", exc.read())
                except (URLError, TimeoutError):
                    self.send(503, "text/plain", b"Training telemetry unavailable")
            elif u.path == '/control':
                state = control.snapshot() if control else dict(state='unavailable')
                self.send(200, 'application/json', json.dumps(dict(**state, supported=bool(control) and error is None,
                    token=control_token, error=error)).encode())
            elif u.path == "/progress":
                def rows(name):
                    file = Path("logs/ppo") / active_run / (name+".jsonl")
                    if not file.exists():
                        return []
                    with file.open(encoding="utf-8") as f:
                        return [json.loads(line) for line in f if line.endswith("\n")]
                data = dict(run=active_run, episodes=rows("episodes") if active_run else [],
                            evaluations=rows("evaluations") if active_run else [])
                self.send(200, "application/json", json.dumps(data).encode())
            elif u.path in ("/frame", "/heartbeat"):
                last_visit = time.monotonic()
                packet = latest
                if error:
                    self.send(503, "text/plain", error.encode())
                elif u.path == "/heartbeat" or packet is None:
                    self.send(204, "text/plain", b"")
                else:
                    self.send(200, "application/octet-stream", packet)
            elif u.path == "/node":
                query = parse_qs(u.query)
                try:
                    if "id" in query:
                        root = int(query["id"][0])
                        if not 0 < root < 2**63:
                            raise ValueError()
                        i = int(np.searchsorted(ids, root))
                        if i == len(ids) or int(ids[i]) != root:
                            raise ValueError()
                    else:
                        i = int(query["i"][0])
                    if not 0 <= i < len(ids):
                        raise ValueError()
                except (KeyError, ValueError):
                    self.send(404, "text/plain", b"Neuron not found")
                    return
                row = dict(i=i, id=str(ids[i]), label=labels[i])
                for name, (ptr, idx, weight) in graph.items():
                    sl = slice(ptr[i], ptr[i+1])
                    row[name] = np.column_stack((idx[sl], weight[sl])).tolist()
                self.send(200, "application/json", json.dumps(row).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            if (self.headers.get('Host') != f'127.0.0.1:{args.port}' or
                    self.headers.get('Origin') != f'http://127.0.0.1:{args.port}'):
                self.send_error(403)
                return
            if self.path != '/control':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 128 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected a small JSON command')
                body = self.rfile.read(size)
                command = json.loads(body)
                if set(command) != {'command'} or command['command'] not in ('pause', 'step', 'play'):
                    raise ValueError('Unknown control command')
            except (ValueError, TypeError):
                self.send(400, 'text/plain', b'Invalid control command')
                return
            if args.observe:
                try:
                    request = Request('http://127.0.0.1:47652/control', data=body, headers={
                        'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:47652',
                        'X-NeonFly-Control': self.headers.get('X-NeonFly-Control', '')}, method='POST')
                    with urlopen(request, timeout=4) as response:
                        self.send(response.status, 'application/json', response.read())
                except HTTPError as exc:
                    self.send(exc.code, 'text/plain', exc.read())
                except (URLError, TimeoutError):
                    self.send(503, 'text/plain', b'Training controller unavailable')
                return
            if not secrets.compare_digest(self.headers.get('X-NeonFly-Control', ''), control_token):
                self.send_error(403)
                return
            if not control or error:
                self.send(503, 'text/plain', b'Training controller unavailable')
                return
            try:
                state = control.command(command['command'])
                self.send(200, 'application/json', json.dumps(state).encode())
            except ValueError as exc:
                self.send(409, 'text/plain', str(exc).encode())

    class LocalServer(ThreadingHTTPServer):
        slots = threading.BoundedSemaphore(8)

        def process_request(self, sock, addr):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(sock)
                return
            sock.settimeout(5)
            super().process_request(sock, addr)

        def process_request_thread(self, sock, addr):
            try:
                super().process_request_thread(sock, addr)
            finally:
                self.slots.release()

    server = LocalServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    if args.observe:
        server.timeout = 1
        with server:
            while time.monotonic()-last_visit <= 120:
                server.handle_request()
        return
    from python.viewer.live import worker
    producer = mp.Process(target=worker, args=(frames, stop, args.device, args.train_run, args.resume,
                                               args.warm_start, args.replay_best, args.continuous,
                                               args.config, args.env_config, control), daemon=True)
    producer.start()
    if args.train_run and sys.platform == "win32":
        import ctypes
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            stop.set()
            raise RuntimeError("Windows refused training sleep inhibition")
    def drain():
        nonlocal latest, error
        while not stop.is_set():
            try:
                kind, data = frames.get(timeout=.5)
                if kind == "frame":
                    note_frame(data)
                    latest = data
                elif kind == "complete":
                    print(data, flush=True)
                    server.shutdown()
                    return
                else:
                    error = data
            except queue.Empty:
                if not producer.is_alive() and error is None:
                    error = "Inference process stopped. Restart the viewer after checking the game."
            if time.monotonic()-last_visit > 120 and (not args.train_run or error):
                server.shutdown()
                return
    threading.Thread(target=drain, daemon=True).start()
    print(f"Viewer http://127.0.0.1:{args.port} | CUDA/CPU producer PID {producer.pid}", flush=True)
    try:
        server.serve_forever()
    finally:
        if args.train_run and sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        stop.set()
        server.server_close()
        producer.join(timeout=5)
        if producer.is_alive():
            producer.terminate()
            producer.join()


if __name__ == "__main__":
    main()

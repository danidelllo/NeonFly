"""Separate model/Unity process. Emits immutable latest-only visual telemetry."""
import json
import base64
import queue
import struct
import time
import traceback
import tomllib
from pathlib import Path

import numpy as np
import torch

from python.environment.ship import ShipConfig, ShipEnv
from python.viewer.prepare import DATA
from python.viewer.policy import demo_weights
from python.vision.connectome import ENCODER, visual_channels, sensory_mapping, resources, preview
from python.vision.range_sensor import RAY_ENCODER, RAY_ENCODERS, LEGACY_RAY_ENCODER, SENSOR_WIDTH, sensor_mapping, sensor_vector


class Reservoir:
    def __init__(self, device, encoder=None):
        load = lambda name: np.load(DATA / (name + ".npy"))
        self.device = device
        if encoder not in (None, ENCODER, *RAY_ENCODERS):
            raise ValueError("unknown fixed encoder")
        self.encoder = encoder
        def tensor(array):
            return torch.from_numpy(array).to(device)
        pointers, indices = load("in_indptr"), load("in_indices")
        counts = load("in_data").astype(np.float32)
        totals = np.zeros(len(pointers), np.float64)
        np.cumsum(counts, out=(sums := np.empty(len(counts), np.float64)))
        totals[1:] = sums[np.maximum(pointers[1:]-1, 0)]
        totals[pointers == 0] = 0
        degree = np.diff(pointers)
        counts *= load("sign")[indices]
        counts /= np.repeat(np.maximum(1, np.diff(totals)).astype(np.float32), degree)
        self.n = len(degree)
        self.weights = torch.sparse_csr_tensor(tensor(pointers), tensor(indices), tensor(counts),
                                               size=(self.n, self.n), device=device)
        visual = load("sensory")
        xy = load("xy")[visual]
        # An artificial projection, not measured visual receptive fields.
        pixels = np.rint((xy-xy.min(0))/np.ptp(xy, axis=0)*83).astype(np.int64)
        self.visual = tensor(visual.astype(np.int64))
        self.pixels = tensor(84*pixels[:, 1]+pixels[:, 0])
        self.outputs = tensor(load("output").astype(np.int64))
        self.readout = tensor(demo_weights(len(self.outputs)))
        self.feature_width = len(self.outputs) * (3 if encoder else 1)
        if encoder:
            visual, slots, resource_neurons = sensory_mapping(load("sensory"))
            self.visual = tensor(visual)
            self.pixels = tensor(slots)
            self.resource_neurons = tensor(resource_neurons)
        if encoder in RAY_ENCODERS:
            neurons, slots = sensor_mapping(encoder)
            self.range_neurons = tensor(neurons)
            self.range_slots = tensor(slots)
        self.clear()

    def clear(self):
        self.state = torch.zeros((self.n, 1), device=self.device)
        self.previous = torch.zeros(len(self.outputs), device=self.device)
        self.average = self.previous.clone()
        self.previous_sensor = None

    @torch.no_grad()
    def advance(self, observation, game=None):
        if self.encoder:
            image = torch.from_numpy(observation.copy()).to(self.device)
            self.visual_maps = visual_channels(image)
            drive = self.visual_maps.flatten()[self.pixels]
            resource_drive = torch.tensor(resources(game), device=self.device).repeat_interleave(64)
            if self.encoder in RAY_ENCODERS:
                self.range_drive = sensor_vector(game, self.previous_sensor)
                expected_width = 32*7+4 if self.encoder == LEGACY_RAY_ENCODER else SENSOR_WIDTH
                if len(self.range_drive) != expected_width:
                    raise ValueError('Installed sensor layout does not match the policy encoder')
                range_drive = torch.from_numpy(self.range_drive).to(self.device)[self.range_slots]
                self.previous_sensor = game
        else:
            image = torch.from_numpy(observation[-1].copy()).to(self.device).flatten()
            drive = image[self.pixels].float() / 127.5 - 1
        for _ in range(3):
            z = torch.sparse.mm(self.weights, self.state)
            z[self.visual, 0] += drive
            if self.encoder:
                z[self.resource_neurons, 0] += resource_drive
            if self.encoder in RAY_ENCODERS:
                z[self.range_neurons, 0] += range_drive
            self.state = 0.4*self.state + 0.6*torch.tanh(z)
        outputs = self.state[self.outputs, 0]
        if not self.encoder:
            return outputs
        difference = outputs - self.previous
        self.average = .9*self.average + .1*outputs
        self.previous = outputs.clone()
        return torch.cat((outputs, difference, self.average))


def terminal_resources(game, previous):
    """Display a dead ship's bars; retain last-known values if it was removed.

    A lethal hit can make health negative. This display-only saturation must
    not weaken validation of observations used for policy decisions.
    """
    result = []
    for k, (current, maximum) in enumerate((('health', 'maxHealth'), ('bullets', 'maxBullets'), ('rockets', 'maxRockets'))):
        value, capacity = game.get(current), game.get(maximum)
        result.append(float(np.clip(2*value/capacity-1, -1, 1))
                      if value is not None and capacity is not None and capacity > 0 else previous[k])
    return result


def worker(messages, stop, device, train_run=None, resume=False, warm_start=None, replay_run=None, continuous=False,
           config_path="configs/reservoir_ppo.toml", env_path="configs/ship.toml", control=None):
    def notify(kind, value):
        try:
            messages.put_nowait((kind, value))
        except queue.Full:
            return False
        return True

    try:
        torch.set_num_threads(2)
        cfg = tomllib.loads(Path(config_path).read_text()) if train_run else {}
        if replay_run:
            replay_saved = torch.load(Path("checkpoints")/replay_run/"best.pt", map_location="cpu", weights_only=True)
            cfg = replay_saved["signature"]["config"]
        model = Reservoir(device, cfg.get("encoder"))
        env = ShipEnv(ShipConfig.load(env_path)) if train_run else ShipEnv()
        if train_run and (env.config.normalize or env.config.resolution != 84 or
                          env.config.rgb != bool(model.encoder)):
            raise ValueError("encoder and observation configuration disagree")
        trainer = None
        replay = None
        eval_rows = []
        if train_run:
            if cfg.get('algorithm') == 'dqn':
                from python.training.reservoir_dqn import Trainer
            else:
                from python.training.reservoir_ppo import Trainer
            trainer = Trainer(train_run, model.feature_width, device, env.config, resume, warm_start, config_path)
            if cfg.get('algorithm') != 'dqn' and len(trainer.returns) == trainer.cfg["episodes_per_update"]:
                trainer.update()
        if replay_run:
            from python.training.reservoir_ppo import ReplayPolicy
            replay = ReplayPolicy(Path("checkpoints")/replay_run/"best.pt", model.feature_width, device)
            env = ShipEnv(ShipConfig(**replay.saved["signature"]["environment"]))
        seq = episodes = dropped = 0
        initialized = False
        def hold():
            from python.communication.observe import request
            state, _ = request(1)
            if not state.get('controlled') or state['timeScale'] != 0:
                raise RuntimeError('Paused game hold was lost')
        if trainer:
            seq, episodes = trainer.steps, trainer.episodes
        while not stop.is_set():
            evaluating = trainer and trainer.should_eval()
            manual_stop = trainer and (trainer.run/"STOP").exists()
            if trainer and (manual_stop or (not continuous and trainer.finished() and not evaluating)):
                trainer.writer.close()
                messages.put(("complete", f"{cfg.get('algorithm', 'ppo').upper()} stopped at update {trainer.updates}, {trainer.steps} steps."))
                return
            if control and initialized and not control.wait(stop, hold):
                return
            obs, info = env.reset()
            initialized = True
            model.clear()
            if trainer and trainer.novelty:
                trainer.novelty.begin_episode(info['game'])
            if not evaluating:
                episodes += 1
            trajectory = []
            elapsed_ticks = 0
            episode_return = 0.
            extrinsic_return = intrinsic_return = 0.
            base_task_return = 0.
            reward_terms = dict(survival=0., damage=0., terminal=0.)
            actions = np.zeros(36, dtype=np.int64)
            while not stop.is_set():
                start = time.perf_counter()
                features = model.advance(obs, info["game"])
                if evaluating or replay:
                    from python.training.reservoir_ppo import ReplayPolicy
                    action = ReplayPolicy.greedy((trainer or replay).head, features)
                elif trainer:
                    action, sample = trainer.decision(features)
                else:
                    action = int((model.readout @ features).argmax())
                activation = model.state[:, 0].cpu().numpy()
                ms = (time.perf_counter()-start)*1000
                seq += 1
                game = info["game"]
                meta = dict(seq=seq, episode=episodes, frame=game["frame"], scene=game["scene"],
                            action=action, inferenceMs=round(ms, 2), device=device, drops=dropped,
                            timestamp=time.time(), side=84, count=model.n)
                if trainer:
                    meta.update(training=True, evaluating=bool(evaluating), update=trainer.updates, run=train_run)
                    meta['algorithm'] = cfg.get('algorithm', 'ppo')
                    if cfg.get('algorithm') == 'dqn':
                        meta.update(epsilon=trainer.epsilon, replaySize=trainer.memory.size)
                    if trainer.novelty and trainer.novelty.coverage_enabled:
                        meta['exploration'] = 'episodic-coverage+rnd'
                    meta['combatRewards'] = bool(env.config.enemy_damage_reward or env.config.enemy_kill_reward)
                if replay:
                    meta.update(replay=True, run=replay_run, update=replay.saved["progress"][2])
                    meta['algorithm'] = cfg.get('algorithm', 'ppo')
                if model.encoder:
                    with torch.no_grad():
                        scores = (trainer or replay).head(torch.nn.functional.layer_norm(features, features.shape))[:36]
                    meta.update(readout="nonlinear", logits=scores.cpu().tolist(),
                                health=game.get("health"), rockets=game.get("rockets"), encoder=model.encoder)
                    # Display-only telemetry, from the same encoder invocation as this action.
                    meta["inputMaps"] = dict(side=42, count=6, encoding="float32-le",
                        data=base64.b64encode(model.visual_maps.cpu().numpy().astype("<f4", copy=False).tobytes()).decode("ascii"))
                    meta["resourceDrive"] = resources(game)
                    if model.encoder in RAY_ENCODERS:
                        meta["rays"] = game["rays"]
                header = json.dumps(meta, separators=(",", ":")).encode()
                packet = struct.pack("<I", len(header))+header+preview(obs).tobytes()+activation.astype("<f4", copy=False).tobytes()
                if not notify("frame", packet):
                    dropped += 1
                if control and not control.wait(stop, hold, consume=True):
                    return
                obs, reward, done, cut, info = env.step(action)
                if control:
                    control.completed()
                extrinsic_return += reward
                base_task_return += info.get('base_task_reward', reward)
                if trainer and trainer.novelty and not evaluating and info["game"].get("rays", {}).get("valid"):
                    safe_bonus = not done and (not trainer.cfg.get('novelty_skip_damage') or
                                                info.get('reward_terms', {}).get('damage', 0.) >= 0)
                    intrinsic = trainer.novelty.reward(info["game"], allow_bonus=safe_bonus)
                    intrinsic_return += intrinsic
                    reward += intrinsic
                episode_return += reward
                actions[action] += 1
                for key, value in info.get("reward_terms", {}).items():
                    reward_terms[key] = reward_terms.get(key, 0.)+value
                if trainer and not evaluating:
                    trajectory.append((sample, reward))
                    elapsed_ticks += info["elapsed_ticks"]
                if done or cut:
                    # Show the final game state before waiting at the next
                    # reset boundary. Do not advance recurrent state twice.
                    if control:
                        game = info['game']
                        seq += 1
                        terminal_meta = {**meta, 'seq': seq, 'frame': game['frame'],
                                         'timestamp': time.time(), 'terminal': True}
                        if model.encoder:
                            terminal_meta['inputMaps'] = dict(side=42, count=6, encoding='float32-le',
                                data=base64.b64encode(visual_channels(torch.from_numpy(obs.copy()).to(device))
                                    .cpu().numpy().astype('<f4', copy=False).tobytes()).decode('ascii'))
                            terminal_meta['resourceDrive'] = terminal_resources(game, meta['resourceDrive'])
                            if model.encoder in RAY_ENCODERS:
                                terminal_meta['rays'] = game['rays']
                        header = json.dumps(terminal_meta, separators=(',', ':')).encode()
                        terminal_packet = struct.pack('<I', len(header))+header+preview(obs).tobytes()+activation.astype('<f4', copy=False).tobytes()
                        while not stop.is_set() and not notify('frame', terminal_packet):
                            time.sleep(.01)
                    if evaluating:
                        eval_rows.append(dict(reward=episode_return, base_task_reward=base_task_return,
                                              combat_reward=reward_terms.get('combat_damage', 0.)+reward_terms.get('combat_kill', 0.),
                                              seconds=info["survived_seconds"],
                                              reason=info["reason"]))
                        if len(eval_rows) == trainer.cfg["eval_episodes"]:
                            trainer.finish_eval(eval_rows)
                            eval_rows = []
                    elif replay:
                        print(f"Replay {episodes}: {info['reason']}, {info['survived_seconds']:.2f}s", flush=True)
                        if episodes >= 3:
                            messages.put(("complete", "Three frozen-policy replay episodes completed."))
                            return
                    elif trainer:
                        if cfg.get('algorithm') == 'dqn':
                            bootstrap = (torch.nn.functional.layer_norm(
                                model.advance(obs, info['game']), features.shape).cpu().numpy()
                                if cut and not done else np.zeros(model.feature_width, np.float32))
                        else:
                            bootstrap = trainer.value(model.advance(obs, info["game"])) if cut and not done else 0.0
                        trainer.end_episode(trajectory, bootstrap, info["reason"], elapsed_ticks,
                                            info["survived_seconds"], dict(reward_terms={**reward_terms, "novelty": intrinsic_return},
                                                extrinsic_reward=extrinsic_return, intrinsic_reward=intrinsic_return,
                                                base_task_reward=base_task_return,
                                                combat_damage_fraction=info.get('combat_damage_fraction', 0.),
                                                combat_kills=info.get('combat_kills', 0),
                                                exploration=dict(trainer.novelty.stats) if trainer.novelty else {},
                                                damage_fraction=info.get("damage_fraction", 0.), actions=actions.tolist()))
                        if cfg.get('algorithm') != 'dqn' and len(trainer.returns) == trainer.cfg["episodes_per_update"]:
                            trainer.update()
                    break
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        while not stop.is_set() and not notify("error", error.splitlines()[-1]):
            time.sleep(.1)
        raise
    finally:
        if control:
            control.close()

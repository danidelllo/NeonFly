"""On-policy PPO on frozen FlyWire features. Update only between complete episodes."""
import hashlib
import json
import os
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from python.viewer.prepare import DATA
from python.training.heads import make_head

# Audited migrations: checkpoint scheduling and action-boundary viewer controls;
# network, optimizer, reward and representation unchanged.
CHECKPOINT_ONLY_PREDECESSORS = {
    "python/training/reservoir_ppo.py": (
        "662447e29c381801e5fd9dd4cff2f8336fa0476a3730223666754b59da1cd1a1",
        "f4f483672144165feca4d1cced8a43e07ef51e0f5b0b9042432bc764954e84a7",
        "1126e44e4e5314b24b3724165fdb79ce633208ca41305ae50ad9468045b1ca8e"),
    "python/viewer/live.py": (
        "a00eb641646fa560b3bd660314781534ac713d4c04d0fe315970290e497ae27f",
        "75938d7e078abf03a54ada3d802a49ac20f07e0aea909a19f46f1bb5f3d0866b",
        "27103d0284c7f82aa2959851a028d85fa5a3cdb7525bcf976ee7166d89cc47e4"),
}


def compatible_signature(saved, current):
    if {k:v for k,v in saved.items() if k != "sources"} != {k:v for k,v in current.items() if k != "sources"}:
        return False
    old, new = saved.get("sources", {}), current.get("sources", {})
    # This exact predecessor lacked the new control helper in its manifest.
    if (new.keys() - old.keys() == {'python/viewer/control.py'} and not old.keys() - new.keys()
            and old.get('python/viewer/live.py') == 'a00eb641646fa560b3bd660314781534ac713d4c04d0fe315970290e497ae27f'
            and old.get('python/training/reservoir_ppo.py') == '662447e29c381801e5fd9dd4cff2f8336fa0476a3730223666754b59da1cd1a1'):
        old = {**old, 'python/viewer/control.py': new['python/viewer/control.py']}
    return old.keys() == new.keys() and all(
        old[k] == new[k] or old[k] in CHECKPOINT_ONLY_PREDECESSORS.get(k, ()) for k in old)


def episode_targets(rewards, values, bootstrap, gamma, decay):
    adv = np.empty(len(rewards), dtype=np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        carry = rewards[i] + gamma * bootstrap - values[i] + gamma * decay * carry
        adv[i] = carry
        bootstrap = values[i]
    return adv, adv + np.asarray(values, dtype=np.float32)


def graph_digest():
    return hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest()


class ReplayPolicy:
    def __init__(self, path, width, device):
        self.saved = torch.load(path, map_location="cpu", weights_only=True)
        sig = self.saved["signature"]
        if sig["graph"] != graph_digest() or sig["width"] != width:
            raise ValueError("policy and connectome do not match")
        self.head = make_head(width, sig["config"], device)
        self.head.load_state_dict(self.saved["head"])
        self.head.eval()

    @staticmethod
    @torch.no_grad()
    def greedy(head, features):
        return int(head(nn.functional.layer_norm(features, features.shape))[:36].argmax())


def evaluation_rank(rows):
    wins = [r for r in rows if r["reason"] == "completion"]
    return (len(wins)/len(rows), -float(np.mean([r["seconds"] for r in wins])) if wins else 0.,
            float(np.mean([r["seconds"] for r in rows])))


class Trainer:
    def __init__(self, run, features, device, env_config, resume=False, warm_start=None,
                 config_path="configs/reservoir_ppo.toml"):
        self.cfg = tomllib.loads(Path(config_path).read_text())
        self.run = Path("logs/ppo") / run
        self.ckpt = Path("checkpoints") / run
        self.ckpt.mkdir(parents=True, exist_ok=bool(resume))
        self.run.mkdir(parents=True, exist_ok=bool(resume))
        self.device = device
        self.signature = dict(config=self.cfg, environment=asdict(env_config), width=features,
                              graph=graph_digest())
        if self.cfg.get("encoder"):
            self.signature["representation"] = self.cfg["encoder"]+";resource64;outputs+delta+ema;leak0.6;substeps3"
            self.signature["sources"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                "python/vision/connectome.py", "python/viewer/live.py", "python/viewer/control.py", "python/training/heads.py",
                "python/training/reservoir_ppo.py", "python/environment/ship.py", "python/communication/observe.py",
                "unity-mod/src/Bridge.cs", "unity-mod/src/GameAdapter.cs")}
            if self.cfg.get("rnd"):
                for name in ("python/vision/range_sensor.py", "python/training/novelty.py", "unity-mod/src/RangeSensor.cs"):
                    self.signature["sources"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        torch.manual_seed(self.cfg["seed"])
        self.head = make_head(features, self.cfg, device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg["learning_rate"], eps=1e-5)
        self.novelty = None
        if self.cfg.get("rnd"):
            from python.training.novelty import GeometryNovelty
            self.novelty = GeometryNovelty(self.cfg, device)
        self.steps = self.episodes = self.updates = 0
        self.last_eval = -1
        self.best = None
        self.batch, self.returns = [], []
        if resume:
            source = self.ckpt/"latest.pt" if resume is True else Path(resume)
            saved = torch.load(source, map_location="cpu", weights_only=True)
            if not compatible_signature(saved["signature"], self.signature):
                raise ValueError("checkpoint graph/configuration mismatch")
            if "optimizer" not in saved:
                raise ValueError("checkpoint is policy-only; choose a resumable episode/latest checkpoint")
            self.head.load_state_dict(saved["head"])
            self.opt.load_state_dict(saved["optimizer"])
            if self.novelty:
                self.novelty.load_state_dict(saved["novelty"])
            self.steps, self.episodes, self.updates = saved["progress"]
            torch.set_rng_state(saved["rng"])
            if device == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"])
            self.batch = [(x.to(device), a.to(device), logp.to(device), adv, ret)
                          for x,a,logp,adv,ret in saved.get("pending_batch", [])]
            self.returns = saved.get("pending_returns", [])
            if "best_rank" in saved:
                self.best = tuple(saved["best_rank"]) if saved["best_rank"] is not None else None
                self.last_eval = saved.get("last_eval", -1)
            elif (self.ckpt/"best.pt").exists():
                self.best = tuple(torch.load(self.ckpt/"best.pt", map_location="cpu",
                                            weights_only=True)["selection"]["rank"])
            if "last_eval" not in saved and (self.run/"evaluations.jsonl").exists():
                self.last_eval = json.loads((self.run/"evaluations.jsonl").read_text().splitlines()[-1])["update"]
            if saved["signature"] != self.signature:
                self.log("checkpoint-migrations", dict(progress=saved["progress"], source=str(source),
                    reason="audited checkpoint/viewer control update; learning configuration unchanged",
                    previous_sources=saved["signature"].get("sources"), current_sources=self.signature.get("sources")))
                (self.run/"config.json").write_text(json.dumps(self.signature, indent=2))
        else:
            if warm_start:
                if self.cfg.get("hidden"):
                    raise ValueError("nonlinear runs start fresh; linear warm start is incompatible")
                saved = torch.load(warm_start, map_location="cpu", weights_only=True)
                if saved["signature"]["graph"] != self.signature["graph"] or saved["signature"]["width"] != features:
                    raise ValueError("incompatible warm start")
                with torch.no_grad():
                    for k in ("weight", "bias"):
                        getattr(self.head, k)[:36].copy_(saved["head"][k][:36])
                (self.run/"parent.json").write_text(json.dumps(dict(path=str(warm_start), progress=saved["progress"])))
            (self.run/"config.json").write_text(json.dumps(self.signature, indent=2))
            self.save_policy("initial", {"reason": "actor before first update"})
        self.writer = SummaryWriter(str(self.run / "tensorboard"))

    @torch.no_grad()
    def decision(self, features):
        x = nn.functional.layer_norm(features, features.shape)
        out = self.head(x)
        policy = Categorical(logits=out[:36])
        a = policy.sample()
        return int(a), (x, a, policy.log_prob(a), float(out[36]))

    @torch.no_grad()
    def value(self, features):
        return float(self.head(nn.functional.layer_norm(features, features.shape))[36])

    def end_episode(self, trajectory, bootstrap, reason, elapsed_ticks, seconds, details=None):
        self.episodes += 1
        rewards = [r for _, r in trajectory]
        values = [sample[3] for sample, _ in trajectory]
        adv, target = episode_targets(rewards, values, bootstrap, self.cfg["gamma"], self.cfg["gae_lambda"])
        self.batch.extend((x, a, logp, v, ret) for ((x, a, logp, _), _), v, ret
                          in zip(trajectory, adv, target, strict=True))
        self.returns.append(sum(rewards))
        self.steps += len(rewards)
        self.log("episodes", dict(episode=self.episodes, steps=self.steps, length=len(rewards),
                                  reward=sum(rewards), reason=reason, elapsed_ticks=elapsed_ticks,
                                  survived_seconds=seconds, **(details or {})))
        print(f"Episode {self.episodes}: {reason}, {seconds:.2f}s, return {sum(rewards):.4f}", flush=True)
        self.writer.add_scalar("episode/return", sum(rewards), self.episodes)
        self.writer.add_scalar("episode/survived_seconds", seconds, self.episodes)
        if reason == "completion" and not (self.ckpt/"first-training-win.pt").exists():
            self.save_policy("first-training-win", {"episode": self.episodes, "reason": reason})

    def save_policy(self, name, selection):
        saved = self.training_state(include_pending=True)
        saved["selection"] = selection
        self.atomic_save(name, saved)

    def training_state(self, include_pending=False):
        return dict(signature=self.signature, head=self.head.state_dict(), optimizer=self.opt.state_dict(),
                    progress=(self.steps, self.episodes, self.updates), rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if self.device == "cuda" else None,
                    best_rank=self.best, last_eval=self.last_eval,
                    pending_batch=[(x.cpu(), a.cpu(), logp.cpu(), float(adv), float(ret))
                                   for x,a,logp,adv,ret in self.batch] if include_pending else [],
                    pending_returns=list(self.returns) if include_pending else [],
                    novelty=self.novelty.state_dict() if self.novelty else None)

    def atomic_save(self, name, saved):
        path = self.ckpt/(name+".tmp")
        torch.save(saved, path)
        os.replace(path, self.ckpt/(name+".pt"))

    def should_eval(self):
        return self.last_eval != self.updates and (self.updates in (0, 1, self.cfg["updates"])
                                                   or self.updates % self.cfg["eval_every_updates"] == 0)

    def finish_eval(self, rows):
        rank = evaluation_rank(rows)
        selected = self.best is None or rank > self.best
        record = dict(update=self.updates, episode=self.episodes, rank=rank, selected=selected,
                      mean_reward=float(np.mean([r["reward"] for r in rows])), trials=rows)
        self.last_eval = self.updates
        if selected:
            self.best = rank
            self.save_policy("best", record)
        self.log("evaluations", record)
        print(f"Evaluation update {self.updates}: {rank}, selected={selected}", flush=True)
        self.last_eval = self.updates

    def log(self, name, row):
        with (self.run / (name+".jsonl")).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")

    def update(self):
        x, a, old, adv, ret = zip(*self.batch)
        x, a, old = torch.stack(x), torch.stack(a), torch.stack(old)
        adv, ret = (torch.as_tensor(np.asarray(arr), device=self.device) for arr in (adv, ret))
        adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
        initial = torch.cat([p.detach().flatten().clone() for p in self.head.parameters()])
        losses, norms = [], []
        kls, entropies, clips = [], [], []
        early_stop = False
        for _ in range(self.cfg["epochs"]):
            for idx in torch.randperm(len(a), device=self.device).split(self.cfg["minibatch"]):
                out = self.head(x[idx])
                dist = Categorical(logits=out[:, :36])
                ratio = (dist.log_prob(a[idx])-old[idx]).exp()
                kl = ((ratio-1)-(dist.log_prob(a[idx])-old[idx])).mean()
                if self.cfg.get("target_kl") and float(kl.detach()) > 1.5*self.cfg["target_kl"]:
                    early_stop = True
                    break
                clip = self.cfg["clip"]
                surrogate = torch.minimum(ratio*adv[idx], ratio.clamp(1-clip, 1+clip)*adv[idx])
                loss = (-surrogate.mean() + 0.5*(ret[idx]-out[:, 36]).square().mean()
                        - self.cfg["entropy_coef"]*dist.entropy().mean())
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg["max_grad_norm"],
                                                error_if_nonfinite=True)
                self.opt.step()
                losses.append(float(loss.detach()))
                norms.append(float(grad))
                kls.append(float(kl.detach()))
                entropies.append(float(dist.entropy().mean().detach()))
                clips.append(float(((ratio-1).abs() > clip).float().mean().detach()))
            if early_stop:
                break
        self.updates += 1
        row = dict(update=self.updates, steps=self.steps, episodes=self.episodes,
                   mean_return=float(np.mean(self.returns)), mean_loss=float(np.mean(losses)),
                   max_grad=max(norms), actor_delta_l2=float((torch.cat([p.detach().flatten() for p in self.head.parameters()])-initial).norm()),
                   approx_kl=float(np.mean(kls)), entropy=float(np.mean(entropies)),
                   clip_fraction=float(np.mean(clips)), kl_early_stop=early_stop,
                   peak_vram_mib=torch.cuda.max_memory_allocated()/2**20 if self.device == "cuda" else 0)
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError("non-finite PPO update")
        saved = self.training_state()
        self.atomic_save("latest", saved)
        if self.episodes > 0 and self.episodes % 20 == 0:
            self.atomic_save(f"episode-{self.episodes:06}", saved)
        self.log("updates", row)
        print(f"Update {self.updates}: return={row['mean_return']:.4f}, entropy={row['entropy']:.3f}, KL={row['approx_kl']:.5f}", flush=True)
        for k, v in row.items():
            self.writer.add_scalar("train/"+k, v, self.steps)
        self.writer.flush()
        self.batch.clear()
        self.returns.clear()

    def finished(self):
        return self.updates >= self.cfg["updates"] or (self.run/"STOP").exists()

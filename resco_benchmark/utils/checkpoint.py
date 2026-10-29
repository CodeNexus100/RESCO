"""Per-episode checkpoints so a training run can be resumed exactly after an interruption.

A checkpoint is taken after an episode's SUMO simulation has been closed (its tripinfo and metrics files are
complete). It holds, for every pfrl DQN inside the agent, pfrl's own snapshot (model, target model, optimizer,
step counters, replay buffer), plus the environment's episode counter and SUMO-seed stream, and the Python, numpy
and torch RNG states. Restoring all of that makes the resumed run continue exactly as an uninterrupted one would.

Layout in cfg.run_path:
    checkpoint_<episode>/      one folder per checkpoint; only the newest complete one is kept
        state.pkl              episode, env state, RNG states
        agent_<name>/          pfrl DQN.save_snapshot() output
        COMPLETE               written last; folders without it are ignored (interrupted mid-write)
    progress.csv               episode, finished_unix_time, episode_seconds (appended every episode)
    final_model_<name>.pt      trained weights (model state_dict), written when the run finishes
"""
import csv
import glob
import os
import pickle
import random
import shutil
import time

import numpy as np

COMPLETE_MARKER = "COMPLETE"


def pfrl_dqns(agent):
    """{name: pfrl DQN} for the RL agents RESCO wraps (IndependentAgent.agents[*].agent, SharedAgent.agent.agent).
    Empty for static controllers."""
    from pfrl.agents import DQN

    found = {}
    if isinstance(getattr(agent, "agents", None), dict):
        for name in sorted(agent.agents):
            inner = getattr(agent.agents[name], "agent", None)
            if isinstance(inner, DQN):
                found[str(name)] = inner
    inner = getattr(getattr(agent, "agent", None), "agent", None)
    if isinstance(inner, DQN):
        found["shared"] = inner
    return found


def _safe_name(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _rng_state():
    import torch

    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    for backend in ("cuda", "xpu"):
        mod = getattr(torch, backend, None)
        if mod is not None and mod.is_available():
            state[backend] = mod.get_rng_state_all()
    return state


def _set_rng_state(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for backend in ("cuda", "xpu"):
        if backend in state:
            getattr(torch, backend).set_rng_state_all(state[backend])


def complete_checkpoints(run_path):
    """[(episode, path)] of checkpoints whose COMPLETE marker exists, oldest first."""
    found = []
    for path in glob.glob(os.path.join(run_path, "checkpoint_*")):
        suffix = os.path.basename(path)[len("checkpoint_"):]
        if suffix.isdigit() and os.path.exists(os.path.join(path, COMPLETE_MARKER)):
            found.append((int(suffix), path))
    return sorted(found)


def save_checkpoint(run_path, episode, agent, env, episode_seconds):
    path = os.path.join(run_path, f"checkpoint_{episode}")
    if os.path.exists(path):
        shutil.rmtree(path)  # leftover from an interrupted attempt at this same episode
    os.makedirs(path)
    for name, dqn in pfrl_dqns(agent).items():
        sub = os.path.join(path, "agent_" + _safe_name(name))
        os.makedirs(sub)
        dqn.save_snapshot(sub)
    state = {
        "episode": episode,
        "env": {
            "cumulative_episode": env.cumulative_episode,
            "sumo_seed_rng": None if env.sumo_seed_rng is None else env.sumo_seed_rng.bit_generator.state,
            "best_reward": env.best_reward,
            "best_episode": env.best_episode,
        },
        "rng": _rng_state(),
    }
    with open(os.path.join(path, "state.pkl"), "wb") as f:
        pickle.dump(state, f)
    with open(os.path.join(path, COMPLETE_MARKER), "w") as f:
        f.write(f"{episode}\n")
    for old_episode, old_path in complete_checkpoints(run_path):
        if old_episode < episode:
            shutil.rmtree(old_path, ignore_errors=True)
    new_file = not os.path.exists(os.path.join(run_path, "progress.csv"))
    with open(os.path.join(run_path, "progress.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "finished_unix_time", "episode_seconds"])
        w.writerow([episode, f"{time.time():.1f}", f"{episode_seconds:.1f}"])


def load_latest_checkpoint(run_path, agent, env):
    """Restore the newest complete checkpoint into agent and env. Returns the episode count it covers (0 if none)."""
    found = complete_checkpoints(run_path)
    if not found:
        return 0
    episode, path = found[-1]
    for name, dqn in pfrl_dqns(agent).items():
        dqn.load_snapshot(os.path.join(path, "agent_" + _safe_name(name)))
    with open(os.path.join(path, "state.pkl"), "rb") as f:
        state = pickle.load(f)
    env.cumulative_episode = state["env"]["cumulative_episode"]
    if state["env"]["sumo_seed_rng"] is not None:
        env.sumo_seed_rng.bit_generator.state = state["env"]["sumo_seed_rng"]
    env.best_reward = state["env"]["best_reward"]
    env.best_episode = state["env"]["best_episode"]
    _set_rng_state(state["rng"])
    return episode


def finalize(run_path, agent):
    """Write the trained weights (small) and remove checkpoint folders (they hold the large replay buffers)."""
    import torch

    for name, dqn in pfrl_dqns(agent).items():
        torch.save(dqn.model.state_dict(), os.path.join(run_path, f"final_model_{_safe_name(name)}.pt"))
    for _, path in complete_checkpoints(run_path):
        shutil.rmtree(path, ignore_errors=True)
    for path in glob.glob(os.path.join(run_path, "checkpoint_*")):
        shutil.rmtree(path, ignore_errors=True)

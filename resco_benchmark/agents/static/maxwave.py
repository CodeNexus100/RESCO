import numpy as np
from resco_benchmark.agents.agent import Agent, IndependentAgent
from resco_benchmark.config.config import config as cfg


class MAXWAVE(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            self.agents[agent_id] = WaveAgent(agent_id)


def signal_pair_map(agent_id):
    """{pair index: green-phase index} for this signal from cfg.pair_to_act_map, or None if the map does not use one.
    Entries that are not int: int (e.g. YAML '0:0' parsed as a string key) are ignored."""
    pair_map = cfg.pair_to_act_map if "pair_to_act_map" in cfg else None
    if not pair_map or agent_id not in pair_map:
        return None
    valid = {p: a for p, a in pair_map[agent_id].items() if isinstance(p, int) and isinstance(a, int)}
    return valid or None


class WaveAgent(Agent):
    def __init__(self, agent_id=None):
        super().__init__()
        # Signals whose green phases don't line up with cfg.phase_pairs (maps with pair_to_act_map) may only choose
        # among their mapped pairs, and act with the mapped green phase. Upstream used the pair index as the phase
        # index, which crashes (KeyError in switch_phase) on cologne3, cologne8, ingolstadt7 and ingolstadt21.
        self.pair_map = signal_pair_map(agent_id)

    def act(self, observation):
        all_press = []
        for pair in cfg["phase_pairs"]:
            left = cfg.directions[pair[0]]
            right = cfg.directions[pair[1]]
            all_press.append(observation[left] + observation[right])

        if self.pair_map is None:
            return np.argmax(all_press)
        candidates = sorted(self.pair_map)  # ascending, so ties go to the lowest pair index, as np.argmax does
        best = max(candidates, key=lambda p: (all_press[p], -p))
        return self.pair_map[best]

    def observe(self, observation, reward, done, info):
        pass

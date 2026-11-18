import os
import random

import numpy as np
from pfrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from pfrl.explorers.epsilon_greedy import select_action_epsilon_greedily

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.utils.utils import permutations_without_rotations, compute_safe_id
import logging

logger = logging.getLogger(__name__)


class FIXED(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for key in obs_act:
            if "mgr" not in key:
                self.agents[key] = FixedAgent(obs_act[key], key)


class FixedAgent(Agent):
    def __init__(self, obs_act, key):
        super().__init__()
        self.agent_id = key
        self.acts = 0
        self.curr_idx = 0
        self.plan = list()
        self.offset = 0
        self.active_phase = 0
        self.active_phase_len = 0
        # Closed loop: the phase the signal served (or was switching to) after the last step, read in observe().
        # None = unknown (start of an episode): act as upstream does.
        self.last_served = None
        num_acts = obs_act[1]

        timings = cfg[key]["fixed_timings"]
        for i in range(num_acts):
            self.plan.append(int(timings[i]))

        phase_orders = np.asarray(list(permutations_without_rotations(range(num_acts))))
        self.phase_order = phase_orders[cfg[key]["fixed_phase_order_idx"]].tolist()

        if "fixed_offset" in cfg[key]:
            self.offset = cfg[key]["fixed_offset"]
            if self.agent_id != "B3":
                logger.warning(
                    "Fixed agent offset not implemented for " + self.agent_id
                )

        self.num_acts = num_acts

    def act(self, _=None):
        # Closed loop: a phase's planned duration counts from the decision at which the signal took up the request.
        # If the last request was not taken up (the environment held the current green for its minimum green, or
        # served a clearing phase first), request the same phase again without counting this decision. Under RESCO's
        # own timing every request is taken up at once, so this never triggers and the plan runs as upstream.
        if self.last_served is not None and self.last_served != self.phase_order[self.active_phase]:
            self.acts += 1
            return self.phase_order[self.active_phase]
        if np.all(np.array(self.plan) == 0):
            self.plan[self.active_phase] = 1  # Ensure that at least one phase is active
        if self.agent_id == "B3" and self.acts == 0:
            # TODO only implemented for 2 signal maps with A3 and B3
            tmp_offset = self.offset
            while tmp_offset > 0:
                if self.active_phase_len > tmp_offset:
                    self.active_phase_len -= tmp_offset
                    tmp_offset = 0
                else:
                    self.active_phase = (self.active_phase - 1) % len(self.plan)
                    swap = self.plan[self.active_phase]
                    swap = swap - tmp_offset
                    if swap < 0:
                        tmp_offset = swap * -1
                    else:
                        self.active_phase_len = swap
                        tmp_offset = 0

        if self.active_phase_len >= np.abs(self.plan[self.active_phase]):
            self.active_phase = (self.active_phase + 1) % len(self.plan)
            while self.plan[self.active_phase] == 0:  # Skip phases with 0 duration
                self.active_phase = (self.active_phase + 1) % len(self.plan)
            self.active_phase_len = 1
        else:
            self.active_phase_len += 1

        self.acts += 1

        return self.phase_order[self.active_phase]

    def __getitem__(self, act):
        return self.plan[self.phase_order[act]]

    def increase_current_phase_length(self, size=1):
        self.plan[self.active_phase] += size

    def decrease_current_phase_length(self, size=1):
        self.plan[self.active_phase] -= size
        if self.plan[self.active_phase] < 0:
            self.plan[self.active_phase] = 0

    def save(self):
        with open(
            os.path.join(
                cfg.run_path, "fixd_{0}.txt".format(compute_safe_id(self.agent_id))
            ),
            "w",
        ) as f:
            f.write(str(self.plan) + "\n")
            f.write(str(self.phase_order) + "\n")
            f.write(str(self.offset) + "\n")

    def observe(self, observation, reward, done, info):
        signal = None
        if not done and isinstance(info, dict) and "environment" in info:
            signal = info["environment"].signals.get(self.agent_id)
        if signal is None:
            self.last_served = None  # episode over (signals are rebuilt on reset) or no signal: unknown
        else:
            self.last_served = signal.next_phase if signal.next_phase is not None else signal.current_phase

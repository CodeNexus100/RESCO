from resco_benchmark.agents.agent import IndependentAgent
from resco_benchmark.agents.static.maxwave import WaveAgent


class MAXPRESSURE(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            self.agents[agent_id] = MaxAgent(agent_id)


class MaxAgent(WaveAgent):
    def act(self, observation):
        return super().act(observation[1:])

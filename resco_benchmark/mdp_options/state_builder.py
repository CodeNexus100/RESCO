from collections import defaultdict

import numpy as np

from resco_benchmark.config.config import config as cfg


INVALID_AGGREGATION_ERR_MSG = "Aggregation function not defined."
INVALID_AGGREGATION_LEVEL_ERR_MSG = "Aggregation level not defined."
VEHICLE_VALUE_ERR_MSG = "Value not found in vehicle variables."


###     Global      ###
def time(signals):
    st = None
    for signal_id in signals:
        signal = signals[signal_id]
        st = signal.sumo.simulation.getTime()
        break
    st_dict = dict()
    for signal_id in signals:
        st_dict[signal_id] = st
    return st_dict


def speed_limit(signals):
    st = list()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            st.append(lane.max_speed)
    st_dict = dict()
    for signal_id in signals:
        st_dict[signal_id] = st
    return st_dict


def lane_length(signals):
    st = list()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            st.append(lane.length)
    st_dict = dict()
    for signal_id in signals:
        st_dict[signal_id] = st
    return st_dict


###     Per Signal  ###
def arrived(signals):
    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        st[signal_id] = signal.observation.arrived
    return st


def departed(signals):
    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        st[signal_id] = signal.observation.departed
    return st


def phase_timers(signals):
    st = defaultdict(list)
    for signal_id in signals:
        signal = signals[signal_id]
        timers = list()
        for i in signal.time_since_phase:
            if i == signal.current_phase:
                timers.append(0)
                timers.insert(0, signal.time_since_phase[i])
            else:
                timers.append(signal.time_since_phase[i])
        st[signal_id] = timers
    return st


def num_lanes(signals):
    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        st[signal_id] = len(signal.lanes)
    return st


###     Directional ###
def pressure(signals):
    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        dir_st = list()
        for direction in signal.lane_sets:
            # Add inbound
            pressure_ = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                pressure_ += lane.queued

            # Subtract downstream
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    pressure_ -= lane.queued
            dir_st.append(pressure_)
        st[signal_id] = dir_st
    return st


###     Per Lane    ####
def effective_running(signals):
    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        running = 0
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            for veh_id in lane.vehicles:
                if lane.vehicles[veh_id].position < lane.max_speed * cfg.step_length:
                    running += 1
        st[signal_id] = running
    return st


def veh_retriever(veh, value):
    if value == "wait":
        return veh.wait
    elif value == "delay":
        return veh.delay
    elif value == "speed":
        return veh.average_speed
    elif value == "acceleration":
        if veh.acceleration > 1:
            return veh.acceleration
        else:
            return 0
    elif value == "deceleration":
        if veh.acceleration < 1:
            return -1 * veh.acceleration
        return 0
    elif value == "position":
        return veh.position
    elif value == "queued":
        if veh.queued:
            return 1
        else:
            return 0
    elif value == "approaching":
        if not veh.queued:
            return 1
        else:
            return 0
    else:
        raise ValueError(VEHICLE_VALUE_ERR_MSG)


def lane_aggregate(signals, aggregate_fn, value, aggregate_level):
    """
    Aggregates vehicle data for traffic signals based on the selected
    aggregation function, data field, and level of aggregation.

    Parameters:
    - signals (dict): A dictionary of signal objects, indexed by signal id.
    - aggregate_fn (str, optional): The aggregation function to apply
      ('average', 'max', or 'min'). Defaults to "average".
    - value (str, optional): The specific field of a vehicle to aggregate,
      such as 'wait'. Defaults to "wait".
    - aggregate_level (str, optional): The granularity of aggregation
      ('lane' or 'direction'). Defaults to "lane".

    Returns:
    - dict: A dictionary with aggregated results for each signal id.

    Raises:
    - NotImplementedError: If an unsupported aggregation function or level
      is provided.
    """
    # TODO Make this aggregate everything in one iteration of vehicles

    # Internal function to allow aggregate at different levels
    def _agg(_lane_id):
        lane = signal.observation.get_lane(_lane_id)
        if aggregate_fn == "average" or aggregate_fn == "sum":
            aggregated = 0
        elif aggregate_fn == "max":
            aggregated = float("-inf")
        elif aggregate_fn == "min":
            aggregated = float("inf")
        else:
            raise NotImplementedError(INVALID_AGGREGATION_ERR_MSG)

        count = len(lane.vehicles)
        for vehicle_id in lane.vehicles:
            vehicle = lane.vehicles[vehicle_id]
            if aggregate_fn == "average" or aggregate_fn == "sum":
                aggregated += veh_retriever(vehicle, value)
            elif aggregate_fn == "max":
                cur_val = veh_retriever(vehicle, value)
                if cur_val > aggregated:
                    aggregated = cur_val
            elif aggregate_fn == "min":
                cur_val = veh_retriever(vehicle, value)
                if cur_val < aggregated:
                    aggregated = cur_val
        if aggregate_fn == "average":
            if count != 0:
                aggregated /= count
        if aggregated == float("inf") or aggregated == float("-inf"):
            aggregated = 0
        return aggregated

    st = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        lane_st = list()
        if aggregate_level == "lane_aggregate":
            for lane_id in signal.lanes:
                lane_st.append(_agg(lane_id))
        elif aggregate_level == "direction_aggregate":
            for direction in signal.lane_sets:
                dir_st = list()
                for lane_id in signal.lane_sets[direction]:
                    dir_st.append(_agg(lane_id))

                if aggregate_fn == "average":
                    lane_st.append(
                        (sum(dir_st) / len(dir_st)) if len(dir_st) > 0 else 0
                    )
                elif aggregate_fn == "max":
                    lane_st.append(max(dir_st) if len(dir_st) > 0 else 0)
                elif aggregate_fn == "min":
                    lane_st.append(min(dir_st) if len(dir_st) > 0 else 0)
                elif aggregate_fn == "sum":
                    lane_st.append(sum(dir_st) if len(dir_st) > 0 else 0)
                else:
                    raise NotImplementedError(
                        INVALID_AGGREGATION_ERR_MSG
                        + f" {aggregate_fn} {value} {aggregate_level}"
                    )
        else:
            raise NotImplementedError(INVALID_AGGREGATION_LEVEL_ERR_MSG)
        st[signal_id] = lane_st
    return st


###     Per Vehicle ###
def vehicle_value(signals, value):
    """
    Collects vehicle-specific values from traffic signals and organizes
    them into a dictionary.

    This function retrieves data about vehicles observed on lanes managed
    by various traffic signals. For each signal, it examines the vehicles
    present in the lanes and extracts detailed values up to a configured
    limit. If fewer vehicles are observed than the limit, the missing
    values are padded with zeros.

    Parameters:
    - signals: A dictionary mapping signal IDs to signal objects, each
      containing lane observations.
    - value: The specific attribute to retrieve from each vehicle.

    Returns:
    - A defaultdict where keys are signal IDs, and values are lists of
      extracted vehicle values, padded to the configured limit if
      necessary.
    """
    st = defaultdict(list)
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            veh_vals = list()
            for vehicle_id in lane.vehicles:
                vehicle = lane.vehicles[vehicle_id]
                veh_vals.append(veh_retriever(vehicle, value))
                if len(veh_vals) == cfg.vehicles_detailed:
                    break
            if len(veh_vals) < cfg.vehicles_detailed:
                veh_vals.extend([0] * (cfg.vehicles_detailed - len(veh_vals)))
            st[signal_id].extend(veh_vals)
    return st


class StateBuilder:
    """
    StateBuilder is a utility class for constructing and managing states
    based on various input signal types, lane attributes, vehicle metrics,
    and aggregation functions. It allows for customizable state
    construction by specifying the information to include and aggregation
    methods.

    Methods:
        __init__: Initializes the StateBuilder object, classifying input
                  state values into categories (signal, lane, vehicle,
                  downstream, aggregation levels) based on predefined
                  functions and allowed state values. Sets defaults for
                  unspecified aggregation levels.
        __call__: Processes input signals to compute states based on
                  included functions and values. Aggregates results from
                  signal, lane, and vehicle-specific computations. Handles
                  downstream propagation of states to signals.

    Raises:
        ValueError: If given state values are invalid or unrecognized.

    Example agent.yaml configuration with all state options:
      vehicles_detailed: 0
      state_builder: [
          "lane_aggregate", "direction_aggregate", "arrived", "departed",
          "phase_timers", "effective_running",
          "average_wait", "average_delay", "average_speed",
          "average_acceleration", "average_deceleration", "max_wait",
          "max_delay", "max_speed", "max_acceleration", "max_deceleration",
          "min_wait", "min_delay", "min_speed", "min_acceleration",
          "min_deceleration", "wait", "delay", "speed", "acceleration",
          "deceleration", "time", "pressure", "downstream_arrived",
          "sum_approaching", "sum_queued",
          "downstream_departed", "downstream_phase_timers",
          "downstream_pressure",
          "downstream_approaching", "downstream_queued",
          "downstream_effective_running", "downstream_average_wait",
          "downstream_average_delay", "downstream_average_speed",
          "downstream_average_acceleration", "downstream_average_deceleration",
          "downstream_max_wait", "downstream_max_delay", "downstream_max_speed",
          "downstream_max_acceleration", "downstream_max_deceleration",
          "downstream_min_wait", "downstream_min_delay", "downstream_min_speed",
          "downstream_min_acceleration", "downstream_min_deceleration",
          "downstream_wait", "downstream_delay", "downstream_speed",
          "downstream_acceleration", "downstream_deceleration",
      ]

    Note:
        - direction_aggregate will not include the default of lane_aggregate.
          Use both if you want both.
        - lane_aggregate is included by default if not specified.
        - vehicles_detailed sets the limit for the number of vehicles to
          include for vehicle level (wait, delay, speed, etc.).
    """

    def __init__(self, state_values):
        vehicle_values = (
            "wait",
            "delay",
            "speed",
            "acceleration",
            "deceleration",
            "position",
            "approaching",
            "queued",
        )
        agg_levels = ("lane_aggregate", "direction_aggregate")
        aggregate_functions = ("average", "max", "min", "sum")

        self.signal_functions = {
            "time": time,
            "speed_limit": speed_limit,
            "lane_length": lane_length,
            "arrived": arrived,
            "departed": departed,
            "phase_timers": phase_timers,
            "num_lanes": num_lanes,
            "pressure": pressure,
        }
        self.lane_functions = {
            "effective_running": effective_running,
        }
        lane_agg_functions = list()
        for agg in aggregate_functions:
            for val in vehicle_values:
                lane_agg_functions.append(f"{agg}_{val}")

        vehicle_functions = [val for val in vehicle_values]

        self.incl_signal = set()
        self.incl_lane = set()
        self.incl_veh = set()
        self.incl_downstream = set()
        self.agg_levels = set()

        # Sort inputs into their functions
        for valu in state_values:
            if valu in self.signal_functions:
                self.incl_signal.add(valu)
            elif valu in self.lane_functions:
                self.incl_lane.add(valu)
            elif valu in vehicle_functions:
                if cfg.vehicles_detailed > 0:
                    self.incl_veh.add(valu)
            elif valu.startswith("downstream_"):
                self.incl_downstream.add(valu[11:])
            elif valu == "direction_aggregate":
                self.agg_levels.add("direction_aggregate")
            elif valu == "lane_aggregate":
                self.agg_levels.add("lane_aggregate")
            else:
                flag = False
                for agg_fn in aggregate_functions:
                    if valu.startswith(f"{agg_fn}_"):
                        flag = True
                        break
                if not flag:
                    raise ValueError(f"Invalid state value: {valu}")

        # Default to lane_aggregation if unspecified
        if len(self.agg_levels) == 0:
            self.agg_levels.add("lane_aggregate")

        # Prepend aggregation level
        for agg_lvl in agg_levels:
            for val in state_values:
                if val in lane_agg_functions:
                    self.incl_lane.add(f"{agg_lvl}+{val}")

        # Propagate aggregation level to downstream states
        if len(self.incl_downstream) != 0:
            for agg_lvl in agg_levels:
                self.incl_downstream.add(agg_lvl)

    def __call__(self, signals):
        call_states, lane_sts, veh_sts = list(), list(), list()
        # Call functions
        for func in sorted(self.incl_signal):  # sorted: set order depends on PYTHONHASHSEED
            call_states.append(self.signal_functions[func](signals))

        for func in sorted(self.incl_lane):
            lvl_spl = func.split("+")
            if len(lvl_spl) == 2:
                val_spl = lvl_spl[1].split("_")
                val = lane_aggregate(
                    signals,
                    aggregate_level=lvl_spl[0],
                    aggregate_fn=val_spl[0],
                    value=val_spl[1],
                )
                lane_sts.append(val)
            else:
                lane_sts.append(self.lane_functions[func](signals))

        for func in sorted(self.incl_veh):
            veh_sts.append(vehicle_value(signals, func))

        # Put results together into states to be sent to each signal
        signal_states = defaultdict(list)
        for signal in signals:
            for signal_st in call_states:
                if type(signal_st[signal]) == list:
                    signal_states[signal].extend(signal_st[signal])
                else:
                    signal_states[signal].append(signal_st[signal])
            for lane_st in lane_sts:
                if type(lane_st[signal]) == list:
                    signal_states[signal].extend(lane_st[signal])
                else:
                    signal_states[signal].append(lane_st[signal])
            for veh_st in veh_sts:
                if type(veh_st[signal]) == list:
                    signal_states[signal].extend(veh_st[signal])
                else:
                    signal_states[signal].append(veh_st[signal])

        # Use StateBuilder to get value of downstream signals TODO maintain names via dicts to avoid StateBuilder call
        if len(self.incl_downstream) != 0:
            for signal_id in signals:
                signal = signals[signal_id]
                dwn_signals = set()
                for direction in signal.lane_sets:
                    for lane_id in signal.lane_sets_outbound[direction]:
                        dwn_signal = signal.out_lane_to_signal_id[lane_id]
                        if dwn_signal in signal.signals:
                            dwn_signals.add(dwn_signal)
                for dwn_signal in sorted(dwn_signals):
                    sb = StateBuilder(self.incl_downstream)
                    dwn_val = list(sb({dwn_signal: signals[dwn_signal]})[dwn_signal])
                    signal_states[signal_id].extend(dwn_val)

        arr_st = dict()
        for signal_id in signals:
            arr_st[signal_id] = np.array(signal_states[signal_id])
        return arr_st


class RewardBuilder:
    """
    RewardBuilder is responsible for processing signals and generating a
    reward based on specified aggregation methods.

    Attributes:
        sb (StateBuilder): An instance of StateBuilder initialized with
        provided state values to be used for reward.

    Methods:
        __init__(state_values):
            Initializes the RewardBuilder with a StateBuilder instance.

        __call__(signals):
            Generates rewards for input signals based on the aggregation
            method specified in the configuration. Supported aggregation
            methods: 'sum', 'max', 'min'. Scales the resulting value by
            the reward_scale defined in the configuration. Raises an error
            if an invalid aggregation method is specified.


    Example agent.yaml configuration:
      reward_builder: ["average_wait"]
      reward_aggregation: "sum"
      reward_scale: -1
    """

    def __init__(self, state_values, reward_aggregation=None):
        if reward_aggregation is None:
            reward_aggregation = cfg.reward_aggregation
        self.sb = StateBuilder(state_values)
        self.reward_aggregation = reward_aggregation

    def __call__(self, signals):
        signal_states = self.sb(signals)
        for signal_id in signals:
            if self.reward_aggregation == "sum":
                signal_states[signal_id] = cfg.reward_scale * sum(
                    signal_states[signal_id]
                )
            elif self.reward_aggregation == "max":
                signal_states[signal_id] = cfg.reward_scale * max(
                    signal_states[signal_id]
                )
            elif self.reward_aggregation == "min":
                signal_states[signal_id] = cfg.reward_scale * min(
                    signal_states[signal_id]
                )
            else:
                raise ValueError(
                    "reward_aggregation parameter must be one of 'sum', 'max', or 'min'"
                )
        return signal_states

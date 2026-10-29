import os
import logging
import random
import csv
import uuid
import xml.etree.ElementTree as ET

from resco_benchmark.config.config import config as cfg

logger = logging.getLogger(__name__)

# This script takes csv files generated from the scrapper.py script and converts them to flow files for SUMO

map_to_signals = {
    "saltlake2_stateXuniversity": ["7243", "7142"],
    "saltlake1A_stateXuniversity": ["7243", None],
    "saltlake1B_stateXuniversity": [None, "7142"],
    "saltlake2_400sX200w": ["7241", "7242"],
    "saltlake1A_400sX200w": ["7241", None],
    "saltlake1B_400sX200w": [None, "7242"],
}

interval = 300  # 5m    Utah ATSPM records with level of precision

# Forgot to log header in scrapper.py
signalId_to_header = {
    "7142": "L,T,R,Total,L,T,R,Total,L,T,TR,Total,L,T,TR,Total".split(","),
    "7241": "L,T,R,Total,L,T,R,Total,L,T,R,Total,L,T,R,Total".split(","),
    "7242": "L,T,TR,Total,L,T,TR,Total,L,T,R,Total,L,T,R,Total".split(","),
    "7243": "L,T,TR,Total,T,TR,Total,L,TR,Total,L,TR,Total".split(","),
}


def add_demand(
    time,
    signal_id,
    direction,
    movement,
    demand,
    routes,
    store,
    vehicle_count,
    sorted_departures,
):
    departure = time * interval
    route_id = signal_id + "_" + direction + movement

    if route_id in routes:
        for vehicle in range(int(demand)):
            depart_time = departure + random.randint(0, interval - 1)
            new_elem = make_sumo_vehicle_element(route_id, depart_time)
            if new_elem is not None:
                vehicle_count += 1
                sorted_departures[depart_time].append(new_elem)

        frac = demand % 1
        if random.random() < frac:
            depart_time = departure + random.randint(0, interval - 1)
            new_elem = make_sumo_vehicle_element(route_id, depart_time)
            if new_elem is not None:
                vehicle_count += 1
                sorted_departures[depart_time].append(new_elem)

    else:  # Demand is requested between signals, processed later in 'adjusted' section
        if time in store:
            store[time].append((signal_id, direction, movement, demand))
        else:
            store[time] = [(signal_id, direction, movement, demand)]
    return vehicle_count


def make_sumo_vehicle_element(route_id, depart_time):
    if depart_time < cfg.start_time:
        return None
    if depart_time >= cfg.end_time:
        return None
    new_elem = ET.Element("vehicle")
    new_elem.set("id", str(uuid.uuid4()))
    new_elem.set("depart", str(depart_time))
    new_elem.set("type", "Car")
    new_elem.set("route", route_id)
    new_elem.set("departLane", "free")
    new_elem.set("departSpeed", "max")
    return new_elem


def generate_flow_from_csv(date_time, flo_file):
    date_time = date_time.strftime("%Y-%m-%d")
    left_signal, right_signal = map_to_signals[cfg.map]
    vehicle_count = 0
    routes = dict()

    tree = ET.parse(flo_file)
    root = tree.getroot()
    for tag in root:
        if tag.tag == "route":
            routes[tag.attrib["id"]] = True

    store = dict()

    sorted_departures = dict()
    for i in range(cfg.start_time, cfg.end_time):
        sorted_departures[i] = list()

    path = os.path.dirname(flo_file)
    n_signals = 0
    for signal_id in map_to_signals[cfg.map]:
        if signal_id is None:
            continue
        try:
            with open(os.path.join(path, signal_id, date_time + ".csv"), "r") as file:
                reader = csv.reader(file)
                header_order = ["E", "W", "N", "S"]
                header = dict()
                direction = header_order.pop(0)
                index = 0
                for movement in signalId_to_header[signal_id]:
                    if movement == "Total":
                        if len(header_order) == 0:
                            continue
                        direction = header_order.pop(0)
                    else:
                        if "TR" in movement:
                            movement = "B"
                        header[index] = direction + movement
                        index += 1
                for time, line in enumerate(reader):
                    for index, demand in enumerate(line):
                        if index == 0:
                            continue  # Time column
                        key = header[index - 1]
                        if key == "Total":
                            continue
                        direction = key[0]
                        movement = key[1]
                        demand = round(int(demand) * cfg.flow)

                        if movement == "B":
                            demand /= 2.0
                            vehicle_count = add_demand(
                                time,
                                signal_id,
                                direction,
                                "T",
                                demand,
                                routes,
                                store,
                                vehicle_count,
                                sorted_departures,
                            )
                            vehicle_count = add_demand(
                                time,
                                signal_id,
                                direction,
                                "R",
                                demand,
                                routes,
                                store,
                                vehicle_count,
                                sorted_departures,
                            )
                        else:
                            vehicle_count = add_demand(
                                time,
                                signal_id,
                                direction,
                                movement,
                                demand,
                                routes,
                                store,
                                vehicle_count,
                                sorted_departures,
                            )
                n_signals += 1
        except FileNotFoundError:
            logger.warning(f"CSV demand file not found for {signal_id} at {date_time}")

    # Handles traffic recorded, but not from an external road
    A3_dest_tot = {"L": 0, "T": 0, "R": 0}
    B3_dest_tot = {"L": 0, "T": 0, "R": 0}
    for time in store:

        A3_orig = {"ET": 0, "NR": 0, "SL": 0}
        B3_orig = {"WT": 0, "NL": 0, "SR": 0}
        A3_dest = {"L": 0, "T": 0, "R": 0}
        B3_dest = {"L": 0, "T": 0, "R": 0}

        for signal_id, direction, movement, demand in store[time]:
            if signal_id == right_signal:
                if direction == "E":
                    A3_dest[movement] += demand
                    A3_dest_tot[movement] += demand
                else:
                    B3_orig[direction + movement] += demand
            elif signal_id == left_signal:
                if direction == "W":
                    B3_dest[movement] += demand
                    B3_dest_tot[movement] += demand
                else:
                    A3_orig[direction + movement] += demand
            else:
                raise Exception("shouldnt happen")

        total = 0
        for key in A3_dest:
            total += A3_dest[key]
        for key in A3_dest:
            A3_dest[key] = A3_dest[key] / total if total != 0 else 0
        adjusted_A3 = {}
        for key in A3_orig:
            for direction in A3_dest:
                adjusted_A3[key + direction] = A3_orig[key] * A3_dest[direction]

        total = 0
        for key in B3_dest:
            total += B3_dest[key]
        for key in B3_dest:
            B3_dest[key] = B3_dest[key] / total if total != 0 else 0
        adjusted_B3 = {}
        for key in B3_orig:
            for direction in A3_dest:
                adjusted_B3[key + direction] = B3_orig[key] * B3_dest[direction]

        for key in adjusted_A3:
            vehicle_count = add_demand(
                time,
                left_signal,
                "",
                key,
                adjusted_A3[key],
                routes,
                store,
                vehicle_count,
                sorted_departures,
            )

        for key in adjusted_B3:
            vehicle_count = add_demand(
                time,
                right_signal,
                "",
                key,
                adjusted_B3[key],
                routes,
                store,
                vehicle_count,
                sorted_departures,
            )

    for i in range(cfg.start_time, cfg.end_time):
        for vehicle in sorted_departures[i]:
            root.append(vehicle)

    # Write xml to file
    # Write into the run folder, never the scenario folder
    fp = os.path.join(cfg.run_path, cfg.uuid + ".flo.xml")
    tree.write(fp)
    return fp, vehicle_count

import xml.etree.ElementTree as ET
import numpy as np
import copy
import os

from resco_benchmark.config.config import config as cfg


# TODO rewrite
def generate_additional_flow():
    map_path = os.path.join(os.path.dirname(cfg.network))
    fractional = cfg.flow % 1
    multiple = int(cfg.flow)

    try:
        tree = ET.parse(os.path.join(map_path, cfg.map + ".rou.xml"))
        root = tree.getroot()

        # Handling flows >= 2.0
        if fractional != 0 and multiple != 0:
            multiple = multiple + 1

        i = 0
        removals = []
        total_vehicles = 0
        for child in root:
            if child.tag == "trip" or child.tag == "vehicle":
                child.set("departLane", "free")
                total_vehicles += 1
                if fractional != 0 and multiple == 0:
                    if np.random.random() > fractional:
                        removals.append(child)
                    continue
                if "_v" in child.attrib["id"]:
                    continue
                if fractional != 0:
                    if np.random.random() > fractional:
                        i += 1
                        continue
                for j in range(1, multiple):
                    new_child = copy.deepcopy(child)

                    new_child.attrib["id"] = new_child.attrib["id"] + "_v" + str(j)
                    root.insert(i + 1, new_child)
                    total_vehicles += 1
                    i += 1
            i += 1

        if fractional != 0:
            if multiple != 0:
                multiple = multiple - 1
            else:
                for child in removals:
                    root.remove(child)
                    total_vehicles -= 1

        # Write into the run folder, never the scenario folder
        new_route = os.path.join(cfg.run_path, cfg.uuid + ".rou.xml")
        tree.write(new_route)

    except FileNotFoundError:
        tree = ET.parse(os.path.join(map_path, cfg.map + ".flo.xml"))
        myroot = tree.getroot()

        total_vehicles = 0
        for interval in myroot.iter("interval"):
            for flow in interval.iter("flow"):
                new_flow = round(int(flow.attrib["number"]) * (multiple + fractional))
                flow.set("number", str(new_flow))
                total_vehicles += new_flow

        new_route = os.path.join(cfg.run_path, cfg.uuid + ".flo.xml")
        tree.write(new_route)

    return new_route, total_vehicles

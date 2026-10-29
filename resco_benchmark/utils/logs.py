import os
import shutil
import logging
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

from resco_benchmark.config.config import config as cfg, hash_config

logger = logging.getLogger(__name__)


# target_dir bypasses default of running UUID
# no_compress forces compression skip regardless of config parameters
def parse_logs(target_dir=None, no_compress=False):
    if target_dir is not None:
        # When called from util scripts config can't be set for each run
        uuid = target_dir[target_dir.rfind(os.sep) :].replace(os.sep, "")
        # Run name is two directories up from the UUID
        run_name = (
            target_dir[: target_dir[: target_dir.rfind(os.sep)].rfind(os.sep)]
            + uuid[-8:]
        )
        # Hash the config to get a unique name
        run_path = target_dir
        config_fp = os.path.join(target_dir, "config.json")
        with open(config_fp) as f:
            json_in = json.load(f)
        hashed_name = hash_config(json_in)
    else:
        run_path = cfg.run_path
        hashed_name = cfg.hashed_name
        run_name = cfg.run_name + cfg.uuid[-8:]

    csv_met_map, eps_avgs, results = dict(), dict(), defaultdict(dict)
    for i, met in enumerate(cfg.csv_metrics):
        csv_met_map[met] = i
    for metric in cfg.xml_metrics + cfg.csv_metrics:
        eps_avgs[metric] = list()
        results[metric] = defaultdict(list)

    trip_files_numbers = []
    for file in os.listdir(run_path):
        if file.startswith("tripinfo_") and file.endswith(".xml"):
            number_part = file[len("tripinfo_") : -len(".xml")]
            if number_part.isdigit() and number_part != "0":
                trip_files_numbers.append(int(number_part))
    i = min(trip_files_numbers) if trip_files_numbers else 1
    while True:
        xml_done = parse_xml_log(run_path, i, eps_avgs)
        csv_done = parse_csv_log(run_path, i, eps_avgs, csv_met_map)
        if xml_done or csv_done:
            break
        i += 1

    # Load previous results if they exist to append new results to the same file

    results_fp = os.path.join(cfg.log_dir, run_name + ".json")
    if os.path.exists(results_fp) and cfg.delete_episode_logs and not no_compress:
        with open(results_fp) as f:
            results = json.load(f)

    # Create new JSON log per run_name
    for metric in cfg.xml_metrics + cfg.csv_metrics:
        results[metric][hashed_name].extend(eps_avgs[metric])

    if "home_log" in cfg:
        try:
            resultsh_fp = os.path.join(
                os.environ.get("HOME"), "results", run_name + ".json"
            )
            os.makedirs(os.path.dirname(resultsh_fp), exist_ok=True)
            print("Saving result json to home directory: {}".format(results_fp))
            with open(resultsh_fp, "w") as f:
                json.dump(results, f)
        except Exception as e:
            logger.error("Failed to save results to home directory: {}".format(e))

    with open(results_fp, "w") as f:
        json.dump(results, f)

    if not no_compress:  # TODO  what is no_compress for?
        compress_folder(run_path)
    return eps_avgs


def parse_xml_log(target_dir, i, eps_avgs):
    trip_file_name = os.path.join(target_dir, "tripinfo_{0}.xml".format(i))
    if not os.path.exists(trip_file_name):
        return True

    # Deformed XML is sometimes output, don't let it stop the rest of the process
    # Happens when running on active processes
    try:
        tree = ET.parse(trip_file_name)
    except ET.ParseError:
        return True

    # Read SUMO output XMLs
    root = tree.getroot()
    num_trips = 0
    totals = dict()
    for metric in cfg.xml_metrics:
        totals[metric] = 0.0
    for child in root:
        if child.attrib["id"].startswith("ghost"):
            continue
        num_trips += 1
        for metric in cfg.xml_metrics:
            totals[metric] += float(child.attrib[metric])
            if metric == "timeLoss":
                totals[metric] += float(child.attrib["departDelay"])
    if num_trips == 0:
        num_trips = 1

    for metric in cfg.xml_metrics:
        eps_avgs[metric].append(totals[metric] / num_trips)
    return False


# RESCO CSV handling
def parse_csv_log(target_dir, i, eps_avgs, csv_met_map):
    trip_file_name = os.path.join(target_dir, "metrics_{0}.csv".format(i))
    if not os.path.exists(trip_file_name):
        return True
    with open(trip_file_name) as fp:
        num_steps = 0
        totals = dict()
        for metric in cfg.csv_metrics:
            totals[metric] = 0.0
        next(fp)  # Skip header
        for line in fp:
            line = line.split("}")
            num_steps += 1
            for metric in cfg.csv_metrics:
                queues = line[csv_met_map[metric]]
                signals = queues.split(":")
                step_total = 0
                for s, signal in enumerate(signals):
                    if s == 0:
                        continue
                    queue = signal.split(",")
                    queue = float(queue[0])
                    step_total += queue
                step_avg = step_total / s
                totals[metric] += step_avg
        # if num_steps == 0: num_steps = 1   TODO why is this here?
        for metric in cfg.csv_metrics:
            eps_avgs[metric].append(totals[metric] / num_steps)
    return False


def compress_folder(folder):
    if cfg.delete_episode_logs:
        for file in os.listdir(folder):  # Only remove performance logs
            if file.startswith(cfg.uuid):
                continue  # Generated route/flow file, still in use by SUMO
            if file.endswith((".xml", ".csv")):
                # skip most recent file in case it is still being written to
                recent_file = max(
                    [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.endswith(".xml")
                    ],
                    key=os.path.getctime,
                )
                if os.path.join(folder, file) == recent_file:
                    continue
                try:
                    os.remove(os.path.join(folder, file))
                except Exception as e:
                    logger.error("Failed to remove {0}: {1}".format(file, e))
    elif shutil.which("pigz") is not None:
        folder = folder[: folder.rfind(os.sep)]
        cmd = 'nohup tar --remove-files -I pigz -cf "{0}.tar.gz" "{1}" &'.format(
            folder, folder
        )
        subprocess.call(cmd, shell=True)
    else:
        folder = folder[: folder.rfind(os.sep)]
        shutil.make_archive(folder, "zip", folder)

        shutil.rmtree(folder)

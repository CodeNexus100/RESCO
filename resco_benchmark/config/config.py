import shutil
import uuid
import hashlib
import os
import sys
import time
import logging
import json
import re

# import git    # TODO causes a problem with HPRC
from quik_config import find_and_load

# Check if in resco_benchmark folder, if not, change
if not os.getcwd().endswith("resco_benchmark") and "utils" not in sys.argv[0]:
    print("Changing directory to project root")
    os.chdir("..")

config_to_ignore_hashing = [
    "uuid",
    "run_uuid",  # fixed run folder for checkpoint/resume; does not change results
    "checkpoint",
    "log_dir",
    "names_map",
    "hashed_name",
    "run_name",
    "run_path",
    "processors",
    "script_launcher",
    "log_level",
    "save_console_log",
    "compress_results",
    "trials",
    "error_bars",
    "min_max",
    "savgol_smoothing",
    "skip_unnamed",
    "xml_metrics",
    "csv_metrics",
    "optuna_trials",
    "episodes",
    "converged",
    "smoothing",
    "benchmark_path",
    "delete_episode_logs",
    "init_ender",
    "save_model",
    "save_log",
    "home_log",
]


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    # TODO put link to overflow question
    """

    def __init__(self, loggr, level):
        self.logger = loggr
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def hash_config(json_in):
    cp = json_in.copy()
    if "load_model" in cp:
        cp["load_model"] = True  # Strip loaded model name
    for key in config_to_ignore_hashing:
        if key in cp:
            del cp[key]
    return hashlib.sha3_256(str(cp).encode("utf-8")).hexdigest()


def load_config():
    config_path = str(os.path.join(os.path.dirname(__file__), "config.yaml"))
    if "sphinx" in sys.argv[0]:
        info = find_and_load(
            config_path, args=["@grid1x1", "@FIXED", "libsumo:False"], parse_args=False
        )
    else:
        info = find_and_load(config_path, fully_parse_args=True)
    config_build = info.config
    profs = info.selected_profiles

    # Get path of log directory if only folder name given
    benchmark_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_build.benchmark_path = benchmark_path
    config_build.log_dir = os.path.expanduser(config_build.log_dir)
    if not os.path.isabs(config_build.log_dir):
        config_build.log_dir = str(os.path.join(benchmark_path, config_build.log_dir))

    # Remove build files
    for folder in ["build", "dist", ".egg-info"]:
        for path in os.listdir(benchmark_path):
            if path.endswith(folder):
                path = os.path.join(benchmark_path, path)
                shutil.rmtree(path)
                print(f"Deleted: {path}")

    if (
        "utils" in sys.argv[0]
        or "experiment_runner" in sys.argv[0]
        or "-c" == sys.argv[0].strip()
    ):
        return config_build

    # Basic attribute adjustment
    config_build.map = profs[0].replace("@", "")
    config_build.algorithm = profs[1].replace("@", "")
    # run_uuid lets a restarted process reuse its run folder (checkpoint resume); otherwise a fresh uuid
    run_uuid = config_build.get("run_uuid") if "run_uuid" in config_build else None
    config_build.uuid = str(run_uuid) if run_uuid else str(uuid.uuid4())

    # Want dates as simple strings
    if "saltlake" in config_build.map:
        config_build.start_date = str(config_build.start_date)
        config_build.end_date = str(config_build.end_date)
        config_build.peak_date = str(config_build.peak_date)

    # Uniquely identifying JSON config for each experiment
    json_rep = json.dumps(info.config, indent=4)
    json_rep = json.loads(json_rep)
    hashed_name = hash_config(json_rep)
    config_build.hashed_name = hashed_name

    run_name = get_run_name(config_build)
    # repo = git.Repo(benchmark_path)
    run_path = (
        str(
            os.path.join(
                config_build.log_dir,
                run_name,
                "git_FINAL",  # + repo.head.commit.hexsha,
                config_build.uuid,
            )
        )
        + os.sep
    )
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    config_build.run_name = run_name
    config_build.run_path = run_path

    # File to help with experiment naming
    with open(
        os.path.join(config_build.log_dir, run_name, hashed_name + ".hsh"), "w"
    ) as f:
        f.write("{0}: {1}\n".format(hashed_name, run_name))
        f.write(" ".join(sys.argv[1:]) + "\n")
        f.write(json.dumps(json_rep, indent=4))
    # Log config in every experiment
    with open(os.path.join(run_path, "config.json"), "w") as f:
        f.write(json.dumps(json_rep, indent=4))

    computed_config(config_build, benchmark_path)

    logger = logging_init(config_build)

    # Write git diff to file
    # TODO causes a problem with HPRC
    # logger.warning("At git commit: " + repo.head.commit.hexsha)
    # diff_file = os.path.join(run_path, "git_diff.txt")
    # with open(diff_file, "w") as f:
    #     f.write(repo.git.diff(repo.head.commit.tree))

    results_fp = os.path.join(config_build.log_dir, run_name)
    print("Saving results to:", results_fp)

    if "$SLURM_JOBID" in os.environ:
        jobid = os.environ["SLURM_JOBID"]
        logger.info("SLURM job ID: " + jobid)
        # Create file in result with jobid
        with open(os.path.join(run_path, jobid + ".jobid"), "w") as f:
            f.write(jobid + "\n")

    return config_build


def get_run_name(config_build):
    if config_build.hashed_name in config_build.names_map:
        run_name = config_build.names_map[config_build.hashed_name]
    else:
        safe_args = list()

        def strip_nonalpha(word):
            return "".join(list(filter(str.isalnum, word)))

        def format_arg(dirty_arg):
            dirty_arg = dirty_arg.lower()
            if "@" in dirty_arg:  # This indicates the algorithm name
                dirty_arg.replace("@", "")
                dirty_arg = strip_nonalpha(dirty_arg)
                return str(dirty_arg)
            else:
                # Replace underscores with camel case, with first letter of each word lower case
                if ":" not in arg:
                    return dirty_arg
                splitted = arg.split(":")
                dirty_arg = splitted[0]
                value = splitted[1]
                dirty_arg = "".join(
                    [word.capitalize() for word in dirty_arg.split("_")]
                )
                # Lowercase first letter
                dirty_arg = dirty_arg[0].lower() + dirty_arg[1:]

                dirty_arg = strip_nonalpha(dirty_arg)
                value = strip_nonalpha(value)
                return dirty_arg + "@" + value

        map_name = ""
        map_qualifier = ""
        skip = False
        for arg in sys.argv[1:]:
            if "load_model" in arg:
                arg = "load_model:True"  # Strip loaded model name
            for ignore in config_to_ignore_hashing:
                skip = False
                if ignore in arg:
                    skip = True
                    break
            if skip:
                continue
            if config_build.map in arg:
                map_name = strip_nonalpha(arg)
            elif "flow" in arg:  # Shift flow into map name
                map_qualifier = strip_nonalpha(arg)
            elif "curriculum" in arg:
                # Curriculum definition can be too long for file paths
                map_qualifier = strip_nonalpha(arg[:10])
            elif "run_hour" in arg:
                map_qualifier = strip_nonalpha(arg[8:])
            else:
                safe_args.append(format_arg(arg))

        run_name = shorten_string("_".join(safe_args))
        if map_qualifier != "":
            run_name = "{0}_{1}+{2}".format(map_name, map_qualifier, run_name)
        else:
            run_name = "{0}+{1}".format(map_name, run_name)

        run_name += "_" + config_build.hashed_name[-8:]
        dirs = config_build.directions
        directions = dict()
        for i, d in enumerate(dirs):
            directions[d] = i
        config_build.directions = directions
        print(config_build.hashed_name + ": " + run_name)
    return run_name


def shorten_string(word, max_word=70):
    if len(word) > max_word:
        # Find any numbers not in scientific notation in the string and limit to 5 length
        word = re.sub(r"(?<!e)(\d+\.\d+)", lambda x: x.group()[:5], word)
        if len(word) > max_word:
            # Disemvowel
            word = re.sub(r"[aeiouAEIOU]", "", word)
            if len(word) > max_word:
                word = word[:max_word]  # Truncate if still too long
    return word


def logging_init(config_build):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.Formatter.converter = time.gmtime
    log_format = "%(asctime)s:%(name)s:%(lineno)d:%(levelname)-8s: %(message)s"
    dt_format = "%Y%m%dT%H%M%SZ"
    level = getattr(logging, config_build.log_level.upper())
    if config_build.save_console_log:
        logging.basicConfig(
            filename=config_build.run_path + "console.log",
            level=level,
            format=log_format,
            datefmt=dt_format,
        )
        std_logger = logging.getLogger("std")
        sys.stdout = StreamToLogger(std_logger, logging.INFO)
        sys.stderr = StreamToLogger(std_logger, logging.ERROR)
    else:
        logging.basicConfig(level=level, format=log_format, datefmt=dt_format)

    return logging.getLogger(__name__)


def search_load_path(project_root, uuid_2load):
    """Search for the path to uuid_2load directory in all subdirectories of project_root"""
    for root, dirs, files in os.walk(project_root, topdown=False):
        for d in dirs:
            if d == uuid_2load:
                return str(os.path.join(root, d))
    raise FileNotFoundError("Could not find load path for " + uuid_2load)


def computed_config(config_build, benchmark_path):
    # Resolve route/network paths
    env_path = str(
        os.path.join(
            benchmark_path, "resco_benchmark", "environments", config_build.map
        )
    )
    if not os.path.isabs(config_build.route):
        config_build.route = os.path.join(env_path, config_build.route)
    if not os.path.exists(config_build.route):
        raise FileNotFoundError("Route file not found: " + config_build.route)
    if not os.path.isabs(config_build.network):
        config_build.network = os.path.join(env_path, config_build.network)
    if not os.path.exists(config_build.network):
        raise FileNotFoundError("Net file not found: " + config_build.network)

    # Reverse mapping for agents to managers
    management = config_build.get("management")
    if management is not None:
        supervisors = dict()
        for manager in management:
            workers = management[manager]
            for worker in workers:
                supervisors[worker] = manager
        config_build.supervisors = supervisors

    if config_build.load_model is not None:
        config_build.load_model = search_load_path(
            benchmark_path, config_build.load_model
        )


def reload_config():
    reload_cfg = load_config()
    for key in config:
        config[key] = reload_cfg[key]


config = load_config()

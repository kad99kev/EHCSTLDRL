import functools
import pathlib
import shutil
import subprocess

from ehcs.config import parse_config


def build_image():
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            f"transfer-sinergym",
            "-f",
            f"dockerfiles/Dockerfile.sinergym",
            ".",
        ]
    )


def run_container(config_path, run_docker, test=False, controller=False):
    cfg = parse_config(config_path)
    cuda = cfg["cuda"]
    cfg["run_name"] += "_" + str(cfg["seed"])
    save_path = pathlib.Path("runs/" + cfg["run_name"])
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_abs = save_path.absolute()

    config_path = pathlib.Path(config_path)
    shutil.copy(config_path, save_path)
    container_cfg_path = save_path / config_path.name

    algo = None
    if not controller:
        algo = cfg["agent"]["algorithm"]

    if run_docker:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
            ]
            + (["--gpus=all"] if cuda else [])
            + [
                "-v",
                f"{save_path_abs}:/ehcs/{save_path}",
                f"transfer-sinergym",
                "/bin/bash",
                "-c",
                (
                    f"ehcs run -c {container_cfg_path}{' -t' if test else ''}{' -cntr' if controller else ''}{f' -a {algo}' if algo else ''}"
                ),
            ]
        )
    else:
        subprocess.run(
            [
                f"ehcs",
                f"run",
                "-c",
                f"{container_cfg_path}",
            ]
            + (["-t"] if test else [])
            + (["-cntr"] if controller else [])
            + (["-a", algo] if algo else [])
        )

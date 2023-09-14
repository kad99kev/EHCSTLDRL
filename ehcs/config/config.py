import functools

import click
import yaml


def parse_config(filename):
    """
    Reads yaml configuration file.
    Arguments:
        filename: The name of the yaml file.
    """
    with open(filename, "r") as yfile:
        cfg = yaml.load(yfile, Loader=yaml.FullLoader)
    return cfg


# Reference: https://stackoverflow.com/a/52147284/13082658
def run_mode_options(f):
    options = [
        click.option(
            "--config",
            "-c",
            default=None,
            required=True,
            help="Path to configuration file.",
            type=str,
        ),
        click.option(
            "--docker", "-d", is_flag=True, help="Whether to run in a docker container."
        ),
    ]
    return functools.reduce(lambda x, opt: opt(x), options, f)


def run_gym_options(f):
    options = [
        click.option(
            "--config",
            "-c",
            default=None,
            required=True,
            help="Path to configuration file.",
            type=str,
        ),
        click.option(
            "--test", "-t", is_flag=True, help="Whether to run experiment in test mode."
        ),
        click.option(
            "--controller",
            "-cntr",
            is_flag=True,
            help="Whether to run experiment with a rule-based controller agent.",
        ),
        click.option(
            "--algo",
            "-a",
            default=None,
            help="Algorithm to run experiment with.",
            type=str,
        ),
    ]
    return functools.reduce(lambda x, opt: opt(x), options, f)

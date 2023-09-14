import click

from ehcs.config import run_gym_options, run_mode_options
from ehcs.utils import docker_utils


@click.group()
def cli():
    """Command line interface for Transfer Sinergym."""


@cli.command()
def build():
    docker_utils.build_image()


@cli.command()
@run_mode_options
def controller(config, docker):
    docker_utils.run_container(config, docker, controller=True)


@cli.command()
@run_mode_options
def train(config, docker):
    docker_utils.run_container(config, docker, test=False)


@cli.command()
@run_mode_options
def test(config, docker):
    docker_utils.run_container(config, docker, test=True)


@cli.command()
@run_gym_options
def run(config, test, controller, algo):
    import ehcs.agents as agents

    if controller:
        agents.controller.run(config)
    else:
        if algo == "ppo":
            if test:
                agents.rl_agent.ppo.test.run(config)
            else:
                agents.rl_agent.ppo.train.run(config)
        elif algo == "sac":
            if test:
                agents.rl_agent.sac.test.run(config)
            else:
                agents.rl_agent.sac.train.run(config)
        else:
            raise ValueError(f"The {algo.title()} algorithm is not supported!")

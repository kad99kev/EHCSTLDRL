# Enhancing HVAC Control Systems through Transfer Learning with Deep Reinforcement Learning Agents

## Installation

In a conda environment, run the following code.

```
git clone https://github.com/kad99kev/EHCSTLDRL.git
pip install -e .
```

## Running an experiment.
Before running an experiment, the Docker environment needs to be built first. This can be done by running:
```
ehcs build
```

Once the Docker container is built, there are different options available:
1. controller - Will run an experiment using a rule-based controller agent.
2. train - Will train a Deep RL agent.
3. test- Will test a trained Deep RL agent.

The commands can be run as follows:
```
ehcs command_name -c path/to/config
```

Sample configuration files for PPO and SAC are given in `configs/`

Experiment tracking with [Weights and Biases](https://wandb.ai/site) is supported. Enter the information required in a `wandb` section of the configuration file to enable experiment tracking.
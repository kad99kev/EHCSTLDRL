from setuptools import find_packages, setup

requirements = [
    "wandb",
    "gymnasium",
    "torch",
    "tensorboard",
    "sinergym==2.3.2",
]

setup(
    name="transfer-sinergym",
    licence="MIT",
    version="0.1",
    url="https://github.com/kad99kev/EHCSTLDRL",
    author="Kevlyn Kadamala",
    author_email="k.kadamala1@universityofgalway.ie",
    description="Source code for 'Enhancing HVAC Control Systems through Transfer Learning with Deep Reinforcement Learning Agents'.",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pylance",
            "isort==5.10.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "ehcs = ehcs.cli:cli",
        ],
    },
)

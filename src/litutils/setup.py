from setuptools import find_packages, setup

setup(
    name="litutils",
    version="0.1.0",
    description="A top-level framework integrating PyTorch Lightning, Optuna, MLflow, and Pydantic for deep learning projects.",
    author="Jan Duchscherer",
    author_email="jan.duchscherer@hm.edu",
    packages=find_packages(),
)

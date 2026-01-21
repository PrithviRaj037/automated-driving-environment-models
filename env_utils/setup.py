from pathlib import Path

from setuptools import setup, find_packages


def read_requirements() -> list[str]:
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        return [line.strip() for line in requirements_file.read_text().splitlines() if line.strip()]
    return []


base_requirements = ["numpy"]
optional_requirements = read_requirements()

setup(
    name="ad-env-utils",
    version="0.2.0",
    description="Environment modelling utilities for automated driving labs (Lanelet2, calibration, data prep).",
    author="Automated Driving Lab",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=base_requirements + optional_requirements,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
    ],
)

"""Setup script for the suture detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="suture-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for suture detection and analysis using YOLOv12",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/suture-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "suture-train=ml_models.core.training.train:main",
            "suture-detect=ml_models.core.inference.infer:main",
        ],
    },
) 
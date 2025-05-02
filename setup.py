from setuptools import setup, find_packages

setup(
    name="auto-suture-pad",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "python-multipart>=0.0.5",
        "numpy>=1.21.0",
        "mlflow>=2.0.0",
        "redis>=4.0.0",
        "minio>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pylint>=2.12.0",
            "black>=22.0.0",
            "pytest-cov>=2.12.0",
        ]
    },
    author="University of Arizona",
    description="A computer vision-based system for analyzing surgical sutures",
    keywords="computer-vision, medical-imaging, yolo, suture-detection",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
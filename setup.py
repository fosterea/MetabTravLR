from setuptools import setup, find_packages
import os


setup(
    name="SpaceTravLR",
    version="0.1.18",  
    description="Spatially perturbing Transcription factors, Ligands & Receptors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "SpaceTravLR_data": ["*.csv", "*.parquet"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "setuptools<81",
        "networkx",
        "pysal>=24.01",
        "scanpy>=1.10.2",
        "torchvision",
        "pyarrow",
        "enlighten",
        "pyro-ppl",
        "group-lasso",
        "pqdm",
        "magic-impute",
        "wordcloud",
        "easydict",
        "torch",
        "ipykernel",
        "simple-slurm",
        "igraph",
        "genomepy",
        "pybedtools",
        "ipywidgets",
        "commot",
        "numpy==1.26"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="spatial transcriptomics, microniche identification, cell-cell communication",
)


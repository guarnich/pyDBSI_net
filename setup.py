from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dbsi-toolbox-net",
    version="1.0.0",
    author="DBSI Toolbox Contributors",
    description=(
        "Physics-informed neural DBSI parameter estimation — "
        "protocol-conditioned Deep Sets architecture"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guarnich/pyDBSI_toolbox_net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "nibabel>=3.2.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "dbsinet-train=scripts.train_dbsi_net:main",
            "dbsinet-run=scripts.run_dbsi_net:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "diffusion MRI", "DBSI", "deep learning", "physics-informed",
        "neuroimaging", "white matter", "multiple sclerosis",
    ],
)

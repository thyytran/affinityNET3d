from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="affinityNET3d",
    version="0.1.0",
    author="Thy Tran",
    author_email="thytranx@gmail.com",
    description="Point cloud deep learning for protein-ligand binding affinity prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thyytran/affinityNET3d",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "biopython>=1.81",
        "rdkit>=2023.3.1",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.9",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
        ],
        "gpu": [
            "torch-geometric>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-affinity=pointcloud_affinity.train:main",
            "serve-affinity=pointcloud_affinity.api:main",
        ],
    },
    include_package_data=True,
)
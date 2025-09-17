from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="facescore",
    version="1.0.0",
    author="FACEScore Team",
    author_email="",
    description="FACEScore: Fourier Analysis of Cross-Entropy for evaluating open-ended natural language generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/CLCS-SUSTech/FACEScore",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "gpu": [
            "torch>=1.9.0+cu118",  # CUDA version
        ],
    },
    entry_points={
        "console_scripts": [
            # Add command-line tools here if needed
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
    },
    keywords="nlp, evaluation, metrics, language-models, fourier-analysis, cross-entropy",
    project_urls={
        "Bug Reports": "https://github.com/CLCS-SUSTech/FACEScore/issues",
        "Source": "https://github.com/CLCS-SUSTech/FACEScore",
        "Documentation": "https://github.com/CLCS-SUSTech/FACEScore#readme",
    },
)

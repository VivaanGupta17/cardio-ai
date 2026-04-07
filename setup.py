"""CardioAI: Deep Learning for ECG Arrhythmia Detection."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cardio-ai",
    version="0.1.0",
    author="CardioAI Contributors",
    description="Deep learning for 12-lead ECG arrhythmia detection and cardiac event prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cardio-ai",
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.0.0",
        ],
        "ecg": [
            "wfdb>=4.0.0",
            "neurokit2>=0.2.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "ECG", "electrocardiogram", "arrhythmia", "deep learning",
        "cardiac AI", "medical device", "signal processing", "PTB-XL",
    ],
    entry_points={
        "console_scripts": [
            "cardioai-train=scripts.train:main",
            "cardioai-evaluate=scripts.evaluate:main",
            "cardioai-predict=scripts.predict:main",
        ],
    },
)

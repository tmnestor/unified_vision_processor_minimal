"""
Setup configuration for Unified Vision Document Processing Architecture
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="unified-vision-processor",
    version="0.1.0",
    author="Developer",
    author_email="developer@example.com",
    description="Unified vision document processing system for Australian tax documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/developer/unified_vision_processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-vision=vision_processor.cli.unified_cli:main",
            "vision-single=vision_processor.cli.single_document:main",
            "vision-batch=vision_processor.cli.batch_processing:main",
            "vision-compare=vision_processor.cli.model_comparison:main",
            "vision-eval=vision_processor.cli.evaluation_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vision_processor": [
            "prompts/*.yaml",
            "config/*.yaml",
            "banking/*.json",
        ],
    },
    keywords=[
        "computer-vision",
        "document-processing",
        "australian-tax",
        "ato-compliance",
        "internvl",
        "llama-vision",
        "multi-gpu",
        "quantization",
    ],
)

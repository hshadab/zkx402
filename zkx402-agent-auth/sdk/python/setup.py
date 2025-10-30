from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zkx402",
    version="0.1.0",
    author="zkX402 Team",
    author_email="contact@zkx402.com",
    description="Python SDK for zkX402 privacy-preserving authorization using zkML proofs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hshadab/zkx402",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black>=23.0.0"],
    },
)

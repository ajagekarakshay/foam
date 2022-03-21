import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foam-rl",
    version="0.0.1",
    author="Akshay Ajagekar",
    author_email="me@akshayajagekar.com",
    description="Framework Agnostic Modular Reinforcement Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajagekarakshay/foam",
    project_urls={
        "Bug Tracker": "https://www.akshayajagekar.com",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "foam"},
    packages=setuptools.find_packages(where="foam"),
    python_requires=">=3.6",
)
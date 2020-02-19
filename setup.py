from setuptools import setup, find_packages

setup(
    name="cvcore",
    version="0.0.1",
    author="",
    author_email="",
    description="Computer Vision Pytorch-based Toolbox",
    url="",
    packages=find_packages(exclude=("logs", "outputs", "weights",)),
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=["albumentations",
                      "torch", "torchvision",
                      #   "pretrainedmodels",
                      "timm"],
)

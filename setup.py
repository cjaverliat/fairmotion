# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import find_packages, setup

setup(
    name="fairmotion",
    version="0.1.0",
    description="fairmotion is FAIR's library for human motion research",
    url="https://github.com/facebookresearch/fairmotion",
    author="FAIR Pittsburgh",
    author_email="dgopinath@fb.com",
    python_requires='>=3.10',
    install_requires=[
        "black",
        "dataclasses",  # py3.6 backport required by human_body_prior
        "matplotlib",
        "numpy",
        "pillow",
        "pyrender==0.1.39",
        "scikit-learn",
        "scipy",
        "torch==1.13.0",
        "tqdm",
        "pyglet==1.5.27",
        "PyOpenGL",
        "PyOpenGL-accelerate @ git+https://github.com/mcfletch/pyopengl.git@227f9c66976d9f5dadf62b9a97e6beaec84831ca#subdirectory=accelerate",
        "body_visualizer @ git+https://github.com/nghorbani/body_visualizer.git@be9cf756f8d1daed870d4c7ad1aa5cc3478a546c",
        "human_body_prior @ git+https://github.com/nghorbani/human_body_prior.git@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc",
        "opencv-python"
    ],
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

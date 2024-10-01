# setup.py
from setuptools import setup, find_packages

setup(
    name="my_yolo_package",
    version="0.1.0",
    description="A package for YOLOv5 object detection and image processing",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'torch',
        'opencv-python',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

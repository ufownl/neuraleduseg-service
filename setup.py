#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Libraries
from setuptools import setup
from os import path
import codecs


##################################################################
# Variables and Constants
PWD = path.abspath(path.dirname(__file__))
ENCODING = "utf-8"

INSTALL_REQUIRES = []
with codecs.open(path.join(PWD, "requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            INSTALL_REQUIRES.append(iline)

##################################################################
# setup()
setup(
    name="neuralseg",
    version="0.1.0a0",
    description=("Discourse segmentation."),
    license="Apache License 2.0",
    url="https://github.com/PKU-TANGENT/NeuralEDUSeg",
    packages=["neuralseg"],
    install_requires=INSTALL_REQUIRES,
    provides=["neuralseg (0.1.0a0)"],
    scripts=[path.join("scripts", "dsegment")],
    author="Yizhong Wang",
    author_email="yishong@pku.edu.cn",
    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Operating System :: Unix",
                 "Operating System :: MacOS",
                 "Programming Language :: Python :: 3",
                 "Topic :: Text Processing :: Linguistic"],
    keywords="discourse NLP linguistics")

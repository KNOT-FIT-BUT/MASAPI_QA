# -*- coding: UTF-8 -*-
""""
Created on 05.11.21

:author:     Martin Dočekal
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='masapiqa',
    version='0.0.1',
    description='Package for QA in MASAPI.',
    long_description_content_type="text/markdown",
    long_description=README,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        'console_scripts': [
            'masapiqa = masapiqa.__main__:main'
        ]
    },
    author='Martin Dočekal',
    keywords=['dataset', 'QA', 'MASAPI'],
    url='https://github.com/mdocekal/MASAPI_QA',
    python_requires='>=3.7',
    install_requires=[
        "windpyutils~=2.0.12",
        "numpy~=1.21.6",
        "transformers==4.10.1",
        "tqdm",
        "setuptools~=61.2.0",
        "torch~=1.7.1",
        "pyserini~=0.17.1",
        "datasets~=2.4.0",
        "scalingqa @  git+https://github.com/KNOT-FIT-BUT/scalingQA.git@main#egg=scalingqa",
        "jsonschema~=4.16.0",
    ]
)

if __name__ == '__main__':
    setup(**setup_args)

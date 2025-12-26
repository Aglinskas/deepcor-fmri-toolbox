"""Setup script for DeepCor."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='deepcor',
    version='0.1.0',
    author='DeepCor Development Team',
    author_email='',
    description='Deep Learning-based Denoising for fMRI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/deepcor-fmri-toolbox',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'sphinx>=4.0',
        ],
    },
    keywords='fmri neuroimaging denoising deep-learning pytorch vae',
    project_urls={
        'Bug Reports': 'https://github.com/Aglinskas/deepcor-fmri-toolbox/issues',
        'Source': 'https://github.com/Aglinskas/deepcor-fmri-toolbox',
    },
)

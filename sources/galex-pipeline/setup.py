from setuptools import setup, find_packages

setup(
    name='galex_pipeline',
    version='0.2.0',
    packages=find_packages(),
    py_modules=['galex_pipeline_class'],
    python_requires='>=3.9',
    install_requires=['astropy', 'numpy', 'matplotlib', 'reproject', 'astroquery', 'pandas'],
    description='GALEX download, paired cropping and spatial response-weighted stacking',
    author='Akhil Krishna R',
    url='https://github.com/akhilkrishnar0/galex-pipeline',
)

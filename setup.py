from setuptools import setup, find_packages

setup(
    name='sweep_manager',
    version='0.0.1',
    packages=find_packages(),
    description='Sweep manager for machine learning using mlflow and optuna',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='KemenczkyP',
    author_email='peter.kemenczky@nje.hu',
    url='https://gitlab.com/nje-ai-research-center/sweep_manager',
    install_requires=[
        'mlflow==2.10.2',
        'optuna==3.5.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

## pip install wheel
## python setup.py sdist bdist_wheel
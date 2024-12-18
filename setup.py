from setuptools import setup, find_packages

setup(
    name='LLM Toaster',
    version='0.5.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'datasets',
        'tqdm',
        'pathlib',
        'tiktoken', 
        'PyYAML',
    ],
    author='Amjad Majid',
    author_email='amjd.y.majid@gmail.com',
)

from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='UniTok',
    version='2.1.9.3',
    keywords=('token', 'tokenizer', 'bert'),
    description='Unified Tokenizer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT Licence',
    url='https://github.com/Jyonn/UnifiedTokenizer',
    author='Jyonn Liu',
    author_email='i@6-79.cn',
    platforms='any',
    packages=find_packages(),
    install_requires=[
        'smartify==0.0.2',
        'termplot==0.0.2',
        'tqdm',
        'numpy',
        'pandas'
    ],
)

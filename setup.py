from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf8')

setup(
    name='UniTok',
    version='4.4.3',
    keywords=['token', 'tokenizer', 'NLP', 'transformers', 'glove', 'bert', 'llama'],
    description='Unified Tokenizer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT Licence',
    url='https://github.com/Jyonn/UnifiedTokenizer',
    author='Jyonn Liu',
    author_email='liu@qijiong.work',
    platforms='any',
    packages=find_packages(),
    install_requires=[
        'termplot==0.0.2',
        'tqdm',
        'numpy',
        'pandas',
        'transformers',
        'oba',
        'prettytable',
        'rich',
        'fastparquet'
    ],
    entry_points={
        'console_scripts': [
            'unitokv3 = UniTokv3.__main__:main',
            'unidep-upgrade-v4 = UniTokv3.__main__:upgrade',
            'unitok = unitok.__main__:main'
        ]
    }
)

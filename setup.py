from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf8')

setup(
    name='UniTok',
    version='4.0.0b2',
    keywords=['token', 'tokenizer'],
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
    ],
    entry_points={
        'console_scripts': [
            'unitok = UniTok.__main__:main',
            'unidep-upgrade-v4 = UniTok.__main__:upgrade',
            'unitokbeta = unitokbeta.__main__:main'
        ]
    }
)

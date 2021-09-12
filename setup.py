from setuptools import setup, find_packages

setup(
    name='UniTok',
    version='0.0.3',
    keywords=('token', 'tokenizer', 'bert'),
    description='Unified Tokenizer',
    long_description='Parse and tokenize data for downstream usage (e.g. the input of Bert model)',
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
    ],
)

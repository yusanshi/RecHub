import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='rechub-yusanshi',
    version='0.0.1',
    author='yusanshi',
    author_email='meet.leiyu@gmail.com',
    description=
    'A package with implementations of some methods in recommendation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yusanshi/RecHub',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.6',
)

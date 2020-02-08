import setuptools

setuptools.setup(
    name="nut",
    version="0.1.0",
    author="Matthew Farrellee",
    author_email="matt@cs.wisc.edu",
    descrption="A playground for document indexing and search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mattf/nut",
    packages=setuptools.find_packages(),
    package_data={
        "nut": ["nut.yaml"],
    },
    install_requires=[
        'connexion[swagger-ui]',
        'gensim',
        'msgpack',
        'msgpack_numpy',
        'numpy',
        'sklearn',
        'tqdm',
        'redis',
        'minhash',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

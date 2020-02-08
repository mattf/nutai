import setuptools

setuptools.setup(
    name="nutai",
    version="0.2.1",
    author="Matthew Farrellee",
    author_email="matt@cs.wisc.edu",
    descrption="A playground for document indexing and search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mattf/nutai",
    packages=setuptools.find_packages(),
    package_data={
        "nutai": ["nutai.yaml"],
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
        'mfoops',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

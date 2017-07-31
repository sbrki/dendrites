from setuptools import setup


setup(
    name='dendrites',
    version="0.1.3",
    packages=[
        'dendrites'
    ],
    platforms='any',
    zip_safe=True,
    install_requires=[
        'numpy'
    ],
    url='https://github.com/whiteShtef/dendrites',
    license="Apache 2.0",
    author="Stjepan",
    author_email="whiteShtef@users.noreply.github.com",
    description='Neural networks for humans',
    long_description="Dendrites aims to be a straightforward and very "
                     "easy to use neural network tool for Python3.",
    keywords="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)

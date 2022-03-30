from setuptools import setup, find_packages

install_requires = [
    'jax>=0.2.12',
    'jaxlib>=0.1.65',
    'jax-md>=0.1.13',
    'dm-haiku>=0.0.4',
    'sympy',
    'chex',
]

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='jax_dimenet',
    version='1.0.0',
    license='Apache 2.0',
    description='DimeNet++ in Jax.',
    author='Stephan Thaler',
    author_email='stephan.thaler@tum.de',
    packages=find_packages(exclude='notebooks'),
    python_requires='>=3.8',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tummfm/jax-dimenet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)

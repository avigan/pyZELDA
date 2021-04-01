from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyzelda',
    version='1.2',
    description='Zernike wavefront sensor analysis and simulation tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/avigan/pyZELDA',
    author="Arthur Vigan & Mamadou N'Diaye",
    author_email='arthur.vigan@lam.fr, mamadou.ndiaye@oca.eu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='zernike wavefront sensor zelda',
    packages=['pyzelda', 'pyzelda.utils'],
    install_requires=[
        'numpy', 'scipy', 'astropy', 'matplotlib'
    ],
    include_package_data=True,
    package_data={
        'pyzelda': ['instruments/*.ini'],
    },
    zip_safe=False
)

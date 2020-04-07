from setuptools import setup

setup(
    name='pyZELDA',
    version='1.1',
    description='Zernike wavefront sensor analysis and simulation tools',
    url='https://github.com/avigan/pyZELDA',
    author='Arthur Vigan & Mamadou N\'Diaye',
    author_email='arthur.vigan@lam.fr, mamadou.ndiaye@oca.eu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Wavefront Sensing',
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

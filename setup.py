from setuptools import setup

setup(
    name='pyZELDA',
    version='0.1',

    description='Zernike wavefront sensor analysis and simulation tools',
    url='https://github.com/avigan/pyZELDA',
    author='Arthur Vigan & Mamadou N\'Diaye',
    author_email='arthur.vigan@lam.fr, mamadou.ndiaye@oca.eu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Topic :: Wavefront Sensing',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='zernike wavefront sensor zelda',
    packages=['pyzelda', 'pyzelda.utils'],
    install_requires=[
        'numpy', 'scipy', 'astropy', 'matplotlib', 'poppy'
    ],
    include_package_data=True,
    package_data={
        'pyzelda': ['instruments/*.ini'],
    },
    zip_safe=False
)

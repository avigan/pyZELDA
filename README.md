pyZELDA
=======

Introduction
------------

This repository provides analysis code for Zernike wavefront sensors in high-contrast imaging applications. ZELDA stands for *Zernike sensor for Extremely Low-level Differential Aberration* ([N'Diaye et al. 2013](https://ui.adsabs.harvard.edu/#abs/2013A&A...555A..94N/abstract)). It is also the name of the Zernike wavefront sensor implemented in the VLT/SPHERE ([Beuzit et al. 2019](https://ui.adsabs.harvard.edu/abs/2019arXiv190204080B/abstract)) instrument.

The formalism of the Zernike wavefront sensor and the demonstration of its capabilities in the VLT/SPHERE instrument are presented in the following papers:

- [N'Diaye, Dohlen, Fusco & Paul, 2013, A&A, 555, A94](https://ui.adsabs.harvard.edu/#abs/2013A&A...555A..94N/abstract)
- [N'Diaye, Vigan, Dohlen et al., 2016, A&A, 592, A79](https://ui.adsabs.harvard.edu/#abs/2016A&A...592A..79N/abstract)
- [Vigan, N'Diaye, Dohlen et al., 2019, A&A, 629, A11](https://ui.adsabs.harvard.edu/abs/2019A%26A...629A..11V/abstract)

Requirements
------------

The package relies on usual packages for data science and astronomy: [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [matplotlib](https://matplotlib.org/) and [astropy](https://www.astropy.org/).

Installation
------------

The easiest is to install `pyzelda` using `pip`:

```sh
pip install pyzelda
```

Otherwise your can download the current repository and install the package manually:

```sh
cd pyZELDA-master/
python setup.py install
```

Citation
--------

If you use this software, or part it, please reference the code from the Astrophysics Source Code Library ([ASCL](http://ascl.net/)):

- [Vigan & N'Diaye, 2018, ascl:1806.003](https://ui.adsabs.harvard.edu/abs/2018ascl.soft06003V/abstract)

Authors and contributors
------------------------

- Arthur Vigan, [arthur.vigan@lam.fr](mailto:arthur.vigan@lam.fr)
- Mamadou N'Diaye, [mamadou.ndiaye@oca.eu](mailto:mamadou.ndiaye@oca.eu)
- Raphaël Pourcelot, [raphael.pourcelot@oca.eu](mailto:raphael.pourcelot@oca.eu)

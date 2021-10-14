# DySMHO

Data-driven discovery of governing equations for dynamical systems via moving horizon optimization. A description of the methods used, main properties, and numerical case studies can be found [here](https://arxiv.org/abs/2108.00069).


## System requirements 

# Software requirements 
- Python 3.7.3
- [Pyomo](http://www.pyomo.org/) 6.1.2
- [numpy](https://numpy.org/) 1.19.5
- [scipy](https://www.scipy.org/) 1.6.2
- [scikit-learn](https://scikit-learn.org/) 0.24.2
- [statsmodels](https://www.statsmodels.org/stable/index.html) 0.12.2
- [matplotlib](https://matplotlib.org/) 3.4.3
- [GAMS](https://www.gams.com/) (https://www.gams.com/)
	- [CONOPT](http://www.conopt.com/) Nonlinear solver (license is required for large instances)
	- See [here](https://www.markdownguide.org/basic-syntax/) for configuration instructions 

# Hardware requirements 
DySMHO requires only a standard computer with enough RAM to support the in-memory operations. The code has been tested on the following systems:
- macOS Big Sur (version 11.6) 
- Windows 10 Enterprise (version 1909) 

	
## Installation guide 

Download the DySMHO repository and ensure all dependencies and requirements above are met. 
```
git clone https://github.com/Baldea-Group/DySMHO
```

## Demo 

Demonstrations for different dynamical systems are included in the [DySMHO/notebook](https://github.com/Baldea-Group/DySMHO/tree/main/DySMHO/notebook) directory. 

## Instructions for use 

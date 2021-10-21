# DySMHO

Data-driven discovery of governing equations for dynamical systems via moving horizon optimization. A description of the methods used, main properties, and numerical case studies can be found [here](https://arxiv.org/abs/2108.00069).


## System requirements 

### Software requirements and versions tested
- [Python](https://www.python.org/) 3.7.3
- [Pyomo](http://www.pyomo.org/) 6.1.2
- [numpy](https://numpy.org/) 1.19.5
- [scipy](https://www.scipy.org/) 1.6.2
- [scikit-learn](https://scikit-learn.org/) 0.24.2
- [statsmodels](https://www.statsmodels.org/stable/index.html) 0.12.2
- [matplotlib](https://matplotlib.org/) 3.4.3
- [GAMS](https://www.gams.com/)
	- [CONOPT](http://www.conopt.com/) Nonlinear solver (license is required for large instances)
	- See [here](https://www.markdownguide.org/basic-syntax/) for configuration instructions 

### Hardware requirements 
DySMHO requires only a standard computer with enough RAM to support the in-memory operations. The code has been tested on the following systems:
- macOS Big Sur (version 11.6) 
- Windows 10 Enterprise (version 1909) 

	
## Installation guide 

Download the DySMHO repository and ensure all dependencies and requirements above are met. 
```
git clone https://github.com/Baldea-Group/DySMHO
```
Installation time should be under 10 seconds with a standard internet conection. Total file size is 22.1 MB. 

## Demo and instructions for use 

Detailed demonstrations for different dynamical systems are included in the [DySMHO/notebook](https://github.com/Baldea-Group/DySMHO/tree/main/DySMHO/notebook) directory. The notebooks explain in detail the following steps: (1) data generation for dynamical systems examples, (2) DySMHO instance definition, (3) application of DySMHO methods (smoothing, pre-processing, discovery, and validation). The expected run times (in seconds) for selected numerical examples on a satndard computer are given below: 

| System      | Data generation (s) | Disovery (s) | 
| ----------- | ----------- | ----------- |
| Header      | Title       | Title       |
| Paragraph   | Text        | Text        |


The directory also includes scripts used to compare the perfromance of DySMHO agains prior works in the literature. 

All the data generating scrips for the numerical case studies can be found in the [data directory](https://github.com/Baldea-Group/DySMHO/tree/main/DySMHO/data).

All utilities and helper functions used in the main model, as well as the definition of the model class for 2D and 3D systems can be found in the [model directory](https://github.com/Baldea-Group/DySMHO/tree/main/DySMHO/model) 

## License 

**MIT License** 

Copyright (c) 2021 Baldea-Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



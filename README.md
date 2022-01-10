# Invariance encoding in sliced-Wasserstein space for image classification with limited training data

This repository contains the Python language codes for reproducing the results in the paper titled "Invariance encoding in sliced-Wasserstein space for image classification with limited training data" using the subspace enhancement technique used with the Radon cumulative distribution transform nearest subspace (RCDT-NS) classifier. To use this classifier users need to install PyTransKit (Python Transport Based Signal Processing Toolkit) from: https://github.com/rohdelab/PyTransKit.

## Installation of PyTransKit

The library can be installed through pip
```
pip install pytranskit
```
Alternately, one can clone/download the repository from [[github](https://github.com/rohdelab/PyTransKit)] and add the `pytranskit` directory to your Python path.
```python
import sys
sys.path.append('path/to/pytranskit')
```

### Location to the datasets:
1. sammas/g_bme-RohdeLab/Shifat/P6_Polynomials/codeP/data/MNIST
2. sammas/g_bme-RohdeLab/Shifat/P6_Polynomials/codeP/data/AFFNISTb
3. sammas/g_bme-RohdeLab/Shifat/P6_Polynomials/codeP/data/OMNIGLOT
4. sammas/g_bme-RohdeLab/Shifat/P6_Polynomials/codeP/data/SYNTH
5. sammas/g_bme-RohdeLab/Shifat/P6_Polynomials/codeP/data/AFFNISTb_out


Instructions:
1. Create a folder named "data" at the same level as the folder "code"
2. Copy the dataset folders above inside this folder "data"

# Biofuels Flash Point Modeling Using Thermodynamically Consistent Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17201047.svg)](https://doi.org/10.5281/zenodo.17201047)

This repository contains the code used to generate the results of our research paper on predicting the **flash point of biofuel mixtures**. The models implemented are **thermodynamically consistent neural networks**, capable of making predictions while enforcing physical constraints.

---

## Installing Dependencies

Before installing the required dependencies, it is recommended to create a Python virtual environment. From the root of the project, run:

```bash
conda env create -f environment.yml
```

After creating the virtual environment, activate it:

```shell
conda activate sciml_venv
```

## Usage

The ```notebooks/``` directory contains:

- **Three notebooks** corresponding to the case studies described in the paper.
- **One notebook** containing the Exploratory Data Analysis (EDA).

All steps followed during the work can be reproduced using these notebooks. Make sure ```jupyter notebook``` is installed and available on your system.

## Contact

Please feel free to contact us or open an issue/discussion if you have any questions, suggestions, or issues. You can also send an email to [mauriomena@hotmail.com](mailto:mauriomena@hotmail.com) if more convenient for you.
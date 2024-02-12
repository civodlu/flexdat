# flexdat

![https://github.com/civodlu/flexdat/actions/workflows/python-package-conda.yml](https://github.com/civodlu/flexdat/actions/workflows/python-package-conda.yml/badge.svg)


This package provides addons for `torch.utils.data.Dataset` to create flexible data pipelines in the context of biomedical 2D/3D/4D image analysis.

## Install
Installation using conda:

    conda create -n flexdat python=3.10
    conda activate flexdat
    git clone <repo>
    pip install -e flexdat[all]

## Overview

- Relies on SimpleITK to handle volumetric data for all dataset transforms
- Relies on numpy for all dataset operations
- Relies on SimpleITK for DICOM loading


## Development Tools
### Editor: VSCode
[VSCode](https://code.visualstudio.com/) is used for the development of this package. A project workspace can be found in `.vscode/flexdat.code-workspace`.

#### Coverage Gutters extension
`pyproject.toml` Should configure `pytest` to export code coverage. `Gutters` can display the coverage in VSCode:
- Install the [Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) VSCode extension
- run `pytest`
- in VSCode, run `Coverage Gutters: Display Coverage` in the command palette

#### Flake8 extension
install extension ms-python.flake8

#### mypy extension
install ms-python.mypy-type-checker

### Useful commands
Useful commands:

    - build the documentation locally: `sphinx-build flexdat/docs/ docsbuild`
    - run static type check: `mypy flexdat/src/`


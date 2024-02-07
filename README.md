# flexdat

This package provides addons for `torch.utils.data.Dataset` to create flexible data pipelines.

## Install
Installation using conda:

    conda create -n flexdat python=3.10
    conda activate flexdat
    git clone <repo>
    pip install -e flexdat[all]

## Tools
### VSCode: Coverage Gutters
`pyproject.toml` Should configure `pytest` to export code coverage. `Gutters` can display the coverage in VSCode:
- Install the [Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) VSCode extension
- run `pytest`
- in VSCode, run `Coverage Gutters: Display Coverage` in the command palette

### VSCode: Flake8
install extension ms-python.flake8

### VSCode: mypy
install ms-python.mypy-type-checker

## Commands
Useful commands:

    - build the documentation locally: sphinx-build flexdat/docs/ docsbuild
    - run typing: mypy flexdat/src/

## mypy


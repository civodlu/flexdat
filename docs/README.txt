mypy cheatsheet:
https://mypy.readthedocs.io/en/stable/mypy_daemon.html#mypy-daemon

# pragma: no cover
# type: ignore
reveal_type()

dmypy start



Callable[
[
    NamedArg(sitk.Image, 'volume'),
    DefaultNamedArg(str, 'base_name'),
],


flake8 src --output-file ci/flake8.txt

genbadge coverage -i ci/coverage.xml -o ci/coverage.svg
genbadge tests -i ci/junit/junit.xml -o ci/tests.svg
genbadge flake8 -i ci/flake8.txt -o ci/flake.svg

https://github.com/civodlu/flexdat/actions/workflows/python-package-conda.yml/badge.svg
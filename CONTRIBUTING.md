# Contributing to NiaAML
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## Code of Conduct
This project and everyone participating in it is governed by the [NiaAML Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [lukapecnik96@gmail.com](mailto:iztok.fister1@um.si).

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible in the issue using the [🐛 bug report issue template](https://github.com/firefly-cpp/NiaAML/blob/master/.github/ISSUE_TEMPLATE/%F0%9F%90%9B%20bug%20report.md).

### Suggesting Enhancements

Open new issue using the [🚀 feature request template](https://github.com/firefly-cpp/NiaAML/blob/master/.github/ISSUE_TEMPLATE/%F0%9F%9A%80%20feature%20request.md).

### Pull requests

Fill in the [pull request template](.github/pull_request_template.md) and make sure your code is documented.

## Setup development environment

### Requirements

* Poetry: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)

After installing Poetry and cloning the project from GitHub, you should run the following command from the root of the cloned project:

```sh
poetry install
```

All of the project's dependencies should be installed and the project ready for further development. **Note that Poetry creates a separate virtual environment for your project.**

### Development dependencies

List of NiaAML's dependencies:

| Package       | Version | Platform |
|---------------|---------|----------|
| numpy         | ^1.19.1 | All      |
| scikit-learn  | ^1.1.2  | All      |
| niapy         | ^2.0.5  | All      |
| pandas        | ^2.1.1  | All      |

List of development dependencies:

| Package           | Version | Platform |
|-------------------|---------|----------|
| sphinx            | ^3.3.1  | Any      |
| sphinx-rtd-theme  | ^0.5.0  | Any      |
| coveralls         | ^2.2.0  | Any      |
| autoflake         | ^1.4    | Any      |
| black             | ^21.5b1 | Any      |
| pre-commit        | ^2.12.1 | Any      |
| pytest            | ^7.4.2  | Any      |
| pytest-cov        | ^4.1.0  | Any      |

## Development Tasks

### Testing

Manually run the tests:

```sh
$ poetry run coverage run --source=niaaml -m unittest discover -b
```

### Documentation

Build the documentation:

```sh
$ poetry run sphinx-build ./docs ./docs/_build
```

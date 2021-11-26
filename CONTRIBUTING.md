# Contributing to NiaAML
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## Code of Conduct
This project and everyone participating in it is governed by the [NiaAML Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [lukapecnik96@gmail.com](mailto:lukapecnik96@gmail.com).

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible in the [issue template](.github/templates/ISSUE_TEMPLATE.md).

### Suggesting Enhancements

Open new issue using the [feature request template](.github/templates/FEATURE_REQUEST.md).

### Pull requests

Fill in the [pull request template](.github/templates/PULL_REQUEST.md) and make sure your code is documented.

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

| Package      | Version    | Platform |
| ------------ |:----------:|:--------:|
| numpy        | ^1.19.1    | All      |
| scikit-learn | ^0.23.2    | All      |
| niapy        | ^2.0.0rc18 | All      |
| pandas       | ^1.1.4     | All      |

List of development dependencies:

| Package                       | Version | Platform |
| ----------------------------- |:-------:|:--------:|
|sphinx                         | ^3.3.1  | Any      |
|sphinx-rtd-theme               | ^0.5.0  | Any      |
|coveralls                      | ^2.2.0  | Any      |

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
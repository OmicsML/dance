# Contributing to dance

Thank you for your interest in contributing to the `dance` package! We welcome
any kinds of contributions such as introducing new features, implementing bug
fixes or additional unit tests, fixing typos or or improving the documentation
in general.

To get started,

1. Make a fork of the [`dance`](https://github.com/OmicsML/dance) repository
   and create a new branch, e.g., `yourname-yourpatch`, where you will make
   your changes there.
1. After you are done with the new changes and have successfully passed our
   [quality checks](#TODO), make a
   [Pull Request](https://github.com/OmicsML/dance/pulls) for you changes with
   descriptions and reasons about the patch.

## Dev notes

### Dev installation

Install PyDANCE with extra dependencies for dev

```bash
pip install -e ."[dev]"
```

Make sure dependencies have specific pinned versions

```bash
pip install -r requirements.txt
```

Install pre-commit hooks for code quality checks

```bash
pre-commit install
```

### Run tests

Run all tests on current environment using pytest

```bash
pytest -v
```

Run full test from the ground up including environment set up using
[tox](https://tox.wiki/en/latest/) on CPU

```bash
tox -e python3.8-cpu
```

## Building documentation

1. Install `dance` with doc dependencies

   ```bash
   pip install -e ."[doc]"
   ```

1. Build the documentation

   ```bash
   cd docs
   make html
   ```

You can now view the documentation page by opening `docs/build/html/index.html`

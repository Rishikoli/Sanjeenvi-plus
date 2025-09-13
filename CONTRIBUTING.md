# Contributing to Sanjeevni Plus

Thank you for your interest in contributing to Sanjeevni Plus! We welcome contributions from the community to help improve this project.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
  - [Documentation](#documentation)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check [existing issues](https://github.com/your-organization/sanjeevni-plus/issues) to see if the problem has already been reported.

When creating a bug report, please include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, browser version, etc.)

### Suggesting Enhancements

We welcome suggestions for new features and improvements. Please:
1. Check if the enhancement has already been suggested
2. Provide a clear and detailed explanation of the enhancement
3. Explain why this enhancement would be useful
4. Include any relevant screenshots or mockups

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Documentation

Improvements to documentation are always welcome! This includes:
- Fixing typos
- Adding missing information
- Improving clarity
- Translating documentation

## Development Setup

1. Fork and clone the repository
2. Set up a Python virtual environment
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
4. Your pull request will be reviewed by the maintainers.

## Coding Standards

- Follow PEP 8 style guide for Python code
- Write docstrings for all public methods and classes
- Include type hints for better code clarity
- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Keep commit messages clear and descriptive

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

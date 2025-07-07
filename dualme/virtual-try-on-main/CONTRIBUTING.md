# Contributing to DualMe Virtual Try-On

Thank you for your interest in contributing to DualMe Virtual Try-On! We welcome contributions from the community.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU development)
- Git
- Docker (optional)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/dualme-virtual-tryon.git
   cd dualme-virtual-tryon
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 🔧 Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain >90% code coverage

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Create examples for new APIs

## 📝 Contribution Process

### 1. Create an Issue

Before starting work, create an issue describing:
- The problem you're solving
- Your proposed solution
- Any relevant context

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Follow the coding guidelines
- Write tests for your changes
- Update documentation as needed

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: add multi-person try-on support"
git commit -m "fix: resolve GPU memory leak in inference"
git commit -m "docs: update deployment guide for salad.com"
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request with:
  - Clear title and description
  - Reference to related issues
  - Screenshots/examples if applicable

## 🎯 Areas for Contribution

### High Priority
- [ ] Performance optimizations
- [ ] Bug fixes
- [ ] Documentation improvements
- [ ] Test coverage improvements

### Features
- [ ] Multi-person try-on
- [ ] Video try-on
- [ ] 3D visualization
- [ ] Mobile app support
- [ ] Custom model training

### Infrastructure
- [ ] CI/CD improvements
- [ ] Deployment scripts
- [ ] Monitoring and logging
- [ ] Security enhancements

## 🐛 Bug Reports

When reporting bugs, include:

- **Environment**: OS, Python version, GPU model
- **Steps to reproduce**: Clear, minimal example
- **Expected vs actual behavior**
- **Error messages and logs**
- **Screenshots** (if applicable)

Use the bug report template when creating issues.

## 💡 Feature Requests

For feature requests, provide:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought of
- **Additional context**: Any other relevant information

## 📋 Code Review Process

All contributions go through code review:

1. **Automated checks**: Tests, linting, security scans
2. **Manual review**: Code quality, design, documentation
3. **Testing**: Functional testing on different environments
4. **Approval**: At least one maintainer approval required

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is acceptable
- [ ] Security considerations are addressed

## 🏷️ Release Process

We follow semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

Releases are tagged and include:
- Changelog with all changes
- Migration guide (for breaking changes)
- Updated documentation

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Be constructive in feedback
- Respect different perspectives
- Help newcomers

### Be Collaborative
- Share knowledge and resources
- Coordinate on larger features
- Communicate clearly and promptly
- Credit others' contributions

### Be Professional
- Follow the code of conduct
- Keep discussions on-topic
- Use appropriate channels
- Maintain confidentiality when needed

## 🆘 Getting Help

- **Discord**: [Join our community](https://discord.gg/dualme)
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers directly for sensitive issues

## 🎖️ Recognition

Contributors are recognized through:

- **Contributors list** in README
- **Release notes** mentioning significant contributions
- **Special badges** for major contributors
- **Maintainer status** for long-term contributors

## 📚 Resources

- [Python Style Guide](https://pep8.org/)
- [Git Best Practices](https://git-scm.com/book)
- [PyTorch Contributing](https://pytorch.org/docs/stable/community/contribution_guide.html)
- [HuggingFace Guide](https://huggingface.co/docs/transformers/contributing)

---

Thank you for contributing to DualMe Virtual Try-On! 🎉 
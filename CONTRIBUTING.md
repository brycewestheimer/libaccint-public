# Contributing to LibAccInt

Thank you for your interest in contributing to LibAccInt! This document provides
guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/libaccint.git`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes with tests
5. Run tests: `ctest --test-dir build/cpu-debug`
6. Format code: `clang-format -i file.cpp`
7. Commit: `git commit -m "feat: add my feature"`
8. Push and create a pull request

## Development Setup

### Prerequisites

- CMake 3.25+
- C++20 compiler (GCC 11+, Clang 14+)
- Git

### Building

```bash
# Debug build (for development)
cmake --preset cpu-debug
cmake --build --preset cpu-debug

# Run tests
ctest --test-dir build/cpu-debug
```

### Code Formatting

Use clang-format before committing:

```bash
find include src -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i
```

## Commit Guidelines

Use conventional commits:

```
type(scope): description

[optional body]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Request review from maintainers
5. Address feedback

## Code of Conduct

Be respectful and constructive. We welcome contributors of all backgrounds and
experience levels.

## License

By contributing, you agree that your contributions will be licensed under the
BSD 3-Clause License.

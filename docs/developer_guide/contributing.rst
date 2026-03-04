.. _contributing:

Contributing Guide
==================

Thank you for your interest in contributing to LibAccInt! This guide will help
you get started.

Getting Started
---------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and clone**:

   .. code-block:: bash

      git clone https://github.com/yourusername/libaccint.git
      cd libaccint

2. **Install dependencies**:

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt install build-essential cmake git clang-format

      # macOS
      brew install cmake llvm

3. **Build in debug mode**:

   .. code-block:: bash

      cmake --preset cpu-debug
      cmake --build --preset cpu-debug

4. **Run tests**:

   .. code-block:: bash

      ctest --test-dir build/cpu-debug --output-on-failure

IDE Setup
~~~~~~~~~

**VS Code** (recommended):

Install extensions:

- C/C++ (Microsoft)
- CMake Tools
- clangd (for better IntelliSense)

Configuration is provided via ``.vscode/`` settings.

**CLion**:

Open the project directory; CMake presets are auto-detected.

Making Changes
--------------

Branching Strategy
~~~~~~~~~~~~~~~~~~

- ``main`` - Stable release branch
- ``develop`` - Integration branch for next release
- ``feature/*`` - Feature branches
- ``fix/*`` - Bug fix branches
- ``docs/*`` - Documentation branches

Create a feature branch:

.. code-block:: bash

   git checkout develop
   git pull origin develop
   git checkout -b feature/my-feature

Commit Guidelines
~~~~~~~~~~~~~~~~~

Use conventional commit messages:

.. code-block:: text

   type(scope): short description

   [optional body]

   [optional footer]

Types:

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation
- ``test``: Tests
- ``refactor``: Code refactoring
- ``perf``: Performance improvement
- ``chore``: Build/tooling changes

Examples:

.. code-block:: text

   feat(engine): add multi-GPU dispatch support

   Implements multi-GPU work distribution in CudaEngine.
   Tested on dual A100 system with CUDA 12.4.

   Closes #123

.. code-block:: text

   fix(basis): correct normalization for d-type shells

   The contraction normalization factor was missing the
   double factorial term for L > 1.

Pull Request Process
--------------------

1. **Ensure all tests pass**:

   .. code-block:: bash

      ctest --test-dir build/cpu-debug

2. **Format your code**:

   .. code-block:: bash

      # Format C++ code
      find include src -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i

      # Format Python code
      black python/
      isort python/

3. **Update documentation** if needed

4. **Create pull request** against ``develop``

5. **Address review feedback**

PR Requirements:

- All CI checks pass
- At least one approving review
- No merge conflicts
- Documentation updated if applicable

Code Review Checklist
~~~~~~~~~~~~~~~~~~~~~

Reviewers will check:

- [ ] Code follows style guide
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] No performance regressions
- [ ] Error handling is appropriate
- [ ] No memory leaks (ASAN clean)

Development Workflow
--------------------

Adding a Feature
~~~~~~~~~~~~~~~~

1. Create an issue describing the feature
2. Discuss design in the issue
3. Create feature branch
4. Implement with tests
5. Update documentation
6. Submit PR
7. Address review feedback
8. Merge after approval

Fixing a Bug
~~~~~~~~~~~~

1. Create an issue with reproduction steps
2. Create fix branch
3. Write a failing test that reproduces the bug
4. Fix the bug
5. Verify test passes
6. Submit PR

Testing
-------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # All tests
   ctest --test-dir build/cpu-debug

   # Specific test
   ctest --test-dir build/cpu-debug -R test_overlap

   # With verbose output
   ctest --test-dir build/cpu-debug -V

   # With address sanitizer
   cmake --preset cpu-debug-asan
   cmake --build --preset cpu-debug-asan
   ctest --test-dir build/cpu-debug-asan

Writing Tests
~~~~~~~~~~~~~

Use GoogleTest framework:

.. code-block:: cpp

   #include <gtest/gtest.h>
   #include <libaccint/libaccint.hpp>

   TEST(OverlapTest, SameShellIsOne) {
       Shell s_shell(AngularMomentum::S, {0,0,0}, {1.0}, {1.0});
       BasisSet basis({s_shell});
       Engine engine(basis);

       std::vector<Real> S;
       engine.compute(OneElectronOperator(Operator::overlap()), S);

       EXPECT_NEAR(S[0], 1.0, 1e-10);
   }

See :doc:`testing` for detailed testing guidelines.

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

Documentation is built at ``docs/_build/html/index.html``.

Documentation Style
~~~~~~~~~~~~~~~~~~~

- Use reStructuredText for Sphinx docs
- Use Doxygen comments for API docs
- Include code examples
- Keep explanations concise

Getting Help
------------

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@libaccint.org (do not open public issues)

Thank you for contributing!

// commitlint.config.js - Conventional Commits configuration for LibAccInt
// See: https://www.conventionalcommits.org/

module.exports = {
  extends: ['@commitlint/config-conventional'],
  
  rules: {
    // Type must be one of the following
    'type-enum': [
      2,
      'always',
      [
        'feat',     // New feature
        'fix',      // Bug fix
        'docs',     // Documentation only
        'style',    // Code style (formatting, semicolons, etc.)
        'refactor', // Code change that neither fixes nor adds
        'perf',     // Performance improvement
        'test',     // Adding or updating tests
        'build',    // Build system or dependencies
        'ci',       // CI configuration
        'chore',    // Maintenance tasks
        'revert',   // Revert a previous commit
      ],
    ],
    
    // Scope is optional but encouraged
    'scope-enum': [
      1,  // Warning only
      'always',
      [
        'core',      // Core library
        'basis',     // Basis set handling
        'engine',    // Integral engine
        'cuda',      // CUDA backend
        'python',    // Python bindings
        'codegen',   // Code generation
        'tests',     // Test suite
        'docs',      // Documentation
        'ci',        // CI/CD
        'deps',      // Dependencies
      ],
    ],
    
    // Subject (description) rules
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    
    // Header (type + scope + subject) max length
    'header-max-length': [2, 'always', 100],
    
    // Body rules
    'body-leading-blank': [2, 'always'],
    'body-max-line-length': [1, 'always', 100],  // Warning only
    
    // Footer rules
    'footer-leading-blank': [2, 'always'],
  },
  
  // Help text for commit message format
  helpUrl: 'https://www.conventionalcommits.org/',
};

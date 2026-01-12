# Project Guidelines

## Code Style
- Use lowercase function/method names with underscores (snake_case)
- Use PascalCase for class names
- Add type hints to function signatures
- Use NumPy-style docstrings
- Keep comments concise and only where logic isn't self-evident

## Project Structure
- `src/data/` - data loading, preprocessing, feature engineering
- `src/models/` - model implementations
- `src/` - utilities and orchestration scripts
- `notebooks/` - exploration and experimentation

## Conventions
- Use TensorFlow for neural network operations
- Use NumPy for numerical computations
- Prefer editing existing files over creating new ones
- Don't add default values to docstrings (they're visible in the signature)

## Don't
- Don't add emoji to code or comments
- Don't create new files unless explicitly asked
- Don't add excessive error handling for internal code
- Don't over-engineer or add unused abstractions

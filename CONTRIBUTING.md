# Contributing

Contributions are welcome. Here's how to get started:

## Setup

```bash
git clone https://github.com/VivaanGupta17/cardio-ai.git
cd cardio-ai
pip install -e ".[dev]"
```

## Development

- Run tests: `pytest tests/ -v`
- Lint: `ruff check src/`
- Format: `ruff format src/`

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Make sure all tests pass
5. Submit a PR with a clear description

## Code Style

- Type hints on all public functions
- Docstrings for classes and complex functions
- Keep functions focused and under 50 lines where possible

# Makefile — CI/CD и проверка кода
# Только для локальных проверок и CI, не для запуска приложения

SOURCE_DIR = ./app
PYTHON = python3.12

.PHONY: format lint clean audit secrets-check all help security-check deps-check init install update lock

## Format code using Black and isort
format:
	@echo "\033[34mFormatting code...\033[0m"
	poetry run black $(SOURCE_DIR)
	poetry run ruff format $(SOURCE_DIR)
	poetry run ruff check --fix $(SOURCE_DIR)
	@echo "\033[32mFormatting complete!\033[0m"

## Lint code using Flake8 and mypy
lint:
	@echo "\033[34mLinting code...\033[0m"
	-poetry run ruff check $(SOURCE_DIR) || echo "ruff found issues"
	-poetry run pyright || echo "pyright found type issues"
	-poetry run vulture $(SOURCE_DIR) --config pyproject.toml || echo "Vulture found dead code"
	-poetry run bandit -r $(SOURCE_DIR) -c .bandit.yml || echo "Bandit found security issues"
	-poetry run pip-audit || echo "PIP-Audit found security issues"
	@echo "\033[32mLinting complete (errors ignored)\033[0m"

## Clean temporary files and caches (without .venv)
clean:
	@echo "\033[34mCleaning project...\033[0m"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov .benchmarks
	@echo "\033[32mCleaning complete!\033[0m"

## Full clean including .venv (optional)
clean-all: clean
	@echo "\033[34mRemoving virtual environment...\033[0m"
	rm -rf .venv
	@echo "\033[32mVirtual environment removed!\033[0m"

## Security audit of dependencies
security-check:
	@echo "\033[34mRunning security checks...\033[0m"
	poetry run safety check || echo "Safety: vulnerabilities found"
	poetry run pip-audit || echo "PIP-Audit: vulnerabilities found"
	@echo "\033[32mSecurity checks complete!\033[0m"

## Check dependencies health (unused, outdated)
deps-check:
	@echo "\033[34mChecking dependencies...\033[0m"
	poetry run deptry . || echo "Deptry: issues found"
	@echo "\033[32mDependencies check complete!\033[0m"

## Scan for secrets in code
secrets-check:
	@echo "\033[34mScanning for secrets...\033[0m"
	poetry run detect-secrets scan --baseline .secrets.baseline || true
	@echo "\033[32mSecrets check completed!\033[0m"

## Full security audit (dependencies + code)
audit: security-check deps-check secrets-check
	@echo "\033[32mFull security audit completed!\033[0m"

## Run full workflow: clean -> format -> lint
all: clean format lint

## Initialize project with Poetry
init:
	@echo "\033[34mInitializing project...\033[0m"
	poetry config virtualenvs.in-project true
	poetry install --with dev
	@echo "\033[32mProject initialized!\033[0m"

## Install dependencies
install:
	poetry install --with dev

## Update dependencies
update:
	poetry update

## Refresh lock file without updating
lock:
	poetry lock --no-update

## Show this help message
help:
	@echo "\033[34mAvailable commands:\033[0m"
	@echo "  make init         - Initialize project with Poetry"
	@echo "  make install      - Install dependencies"
	@echo "  make update       - Update dependencies"
	@echo "  make lock         - Refresh lock file"
	@echo "  make format       - Format code using Black and isort"
	@echo "  make lint         - Lint code using Flake8 and mypy"
	@echo "  make clean        - Remove temporary files and caches"
	@echo "  make clean-all    - Full clean including .venv (DANGER!)"
	@echo "  make security-check - Check for dependency vulnerabilities"
	@echo "  make deps-check   - Check for unused/outdated dependencies"
	@echo "  make secrets-check - Scan for secrets in code"
	@echo "  make audit        - Run full security audit"
	@echo "  make all          - Run full workflow (clean, format, lint)"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "\033[34mCustom variables:\033[0m"
	@echo "  SOURCE_DIR=...    - Source directory (default: ./app)"
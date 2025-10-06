.PHONY: help format lint test coverage docs clean install

# 默认目标
help:
	@echo "FoloVLLM Development Commands"
	@echo ""
	@echo "make install     - Install dependencies"
	@echo "make format      - Format code with black and isort"
	@echo "make lint        - Run linters (flake8, mypy)"
	@echo "make test        - Run all tests"
	@echo "make coverage    - Run tests with coverage report"
	@echo "make docs        - Generate documentation"
	@echo "make clean       - Clean temporary files"
	@echo ""

# 安装依赖
install:
	pip install -e ".[dev]"

# 代码格式化
format:
	@echo "Running black..."
	black folovllm/ tests/ examples/ --line-length 100
	@echo "Running isort..."
	isort folovllm/ tests/ examples/
	@echo "✅ Code formatted"

# 代码检查
lint:
	@echo "Running flake8..."
	flake8 folovllm/ tests/ --max-line-length 100 --ignore E203,W503
	@echo "Running mypy..."
	mypy folovllm/ --ignore-missing-imports
	@echo "✅ Linting passed"

# 运行测试
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "✅ Tests passed"

# 测试覆盖率
coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=folovllm --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html to view the report"

# 生成文档
docs:
	@echo "Generating API documentation..."
	pdoc --html folovllm -o docs/api/ --force
	@echo "✅ Documentation generated in docs/api/"

# 清理临时文件
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	@echo "✅ Cleaned"

# 运行示例
example:
	@echo "Running basic inference example..."
	python examples/basic_inference.py

# 性能测试
benchmark:
	@echo "Running benchmark..."
	python tests/benchmark/run_benchmark.py

# 检查文档完整性
check-docs:
	@echo "Checking documentation..."
	@for milestone in 0 1 2 3 4 5 6 7; do \
		if [ ! -f "docs/dev/milestone_$$milestone.md" ]; then \
			echo "❌ Missing: docs/dev/milestone_$$milestone.md"; \
		fi; \
	done
	@echo "✅ Documentation check complete"

# 完整 CI 流程
ci: format lint test
	@echo "✅ CI pipeline complete"


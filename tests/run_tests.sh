#!/bin/bash
# tests/run_tests.sh

echo "🧪 Running Fraud Detection Pipeline Tests"
echo "========================================="

# Install test dependencies
echo "📦 Installing dependencies..."
poetry install

# Run unit tests
echo "🔬 Running unit tests..."
poetry run pytest tests/unit -v --cov=data_pipeline --cov=backend

# Run integration tests (if GCP credentials available)
echo "🔗 Running integration tests..."
poetry run pytest tests/integration -v -m "not requires_gcp"

# Run performance tests
echo "⚡ Running performance tests..."
poetry run pytest tests/test_performance.py -v -m "not slow"

# Generate coverage report
echo "📊 Generating coverage report..."
poetry run pytest --cov=data_pipeline --cov=backend --cov-report=html --cov-report=term

# Run linting
echo "🎨 Running code quality checks..."
poetry run black --check data_pipeline backend tests
poetry run isort --check-only data_pipeline backend tests

# Run type checking
echo "🔍 Running type checking..."
poetry run mypy data_pipeline backend

echo "✅ All tests complete!"
echo ""
echo "📈 Coverage report available at: htmlcov/index.html"
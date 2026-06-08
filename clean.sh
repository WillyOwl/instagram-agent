#!/usr/bin/env bash
# clean.sh — Clean up all python, pytest, mypy, and ruff caches/temporary files.

echo "Cleaning up temporary files and cache directories..."

# Remove directory caches
rm -rf .pytest_cache
rm -rf .mypy_cache
rm -rf .ruff_cache

# Find and delete Python pycache directories and compiled files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove coverage files if they exist
rm -f .coverage
rm -rf htmlcov

echo "All caches successfully cleared!"

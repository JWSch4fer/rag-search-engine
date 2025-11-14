
# How to run tests

1) Create a venv and install dependencies:
   pip install -r requirements-test.txt

2) From the project root (same folder as pyproject.toml), run:
   pytest

If you want to run a single file:
   pytest tests/test_utils_preprocess.py::test_preprocess_basics -q

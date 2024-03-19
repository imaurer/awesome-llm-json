ACTIVATE = . ./activate.sh

sync:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip sync requirements.txt
	uv pip install -e "."
	uv pip freeze

pbcopy:
	# copy all code to clipboard for pasting into an LLM
	find . ! -path '*/.*/*' -type f \( -name "*.py" -o -name "*.md" \) -exec tail -n +1 {} + | pbcopy

#----------
# clean
#----------

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .cache
	rm -fr .mypy_cache
	rm -fr .pytest_cache
	rm -f .coverage
	rm -fr htmlcov/
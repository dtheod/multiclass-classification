.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
activate:
	@echo "Activating virtual environment"
	poetry shell

test:
	pytest --no-header -v  

docs_view:
	@echo View API documentation... 
	pdoc src --http localhost:8787

docs_save:
	@echo Save documentation to docs... 
	pdoc src -o docs

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

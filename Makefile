PYTHON = python

.PHONY: help install notebook test clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install project dependencies"
	@echo "  make notebook  - Launch Jupyter Notebook"
	@echo "  make test      - Run project tests"
	@echo "  make clean     - Remove Python cache files"

install:
	$(PYTHON) -m pip install -r requirements.txt

notebook:
	$(PYTHON) -m jupyter notebook

test:
	$(PYTHON) -m pytest tests/

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

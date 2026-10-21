.PHONY: install format train test eval report

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black train.py tests

train:
	python train.py

test:
	pytest -q

eval:
	python -c "from pathlib import Path; print(Path('Results/metrics.txt').read_text())"

report:
	echo "# Model Metrics" > report.md
	cat Results/metrics.txt >> report.md
	echo "\n## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](Results/model_results.png)" >> report.md

# CI/CD for Machine Learning

A beginner-friendly ML lifecycle demo using GitHub Actions, CML, scikit-learn,
Gradio, and Hugging Face Spaces.

## Dataset setup

Download a drug classification CSV with columns such as `Age`, `Sex`, `BP`,
`Cholesterol`, `Na_to_K`, and target column `Drug`. Save it as `Data/drug.csv`.
Use a dataset you have permission to use. This app is an educational demo only.

## Local run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux/WSL: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
pip install -r App/requirements.txt
python App/drug_app.py
```

## Hugging Face deployment setup

1. Create a Gradio Space.
2. Create a Hugging Face write token and add it to GitHub repository Secrets as `HF`.
3. Add a GitHub repository variable named `HF_SPACE_ID` with value `your-username/your-space-name`.
4. Push to `main`. CI trains and uploads artifacts; CD deploys after successful CI.

## Important

This is a learning project, not a medical tool. The demo dataset is small and model
performance must not be interpreted as clinical evidence.

- Assignment presentation for the DevOps course.

## CI/CD validation

This project automatically trains and evaluates its scikit-learn model
using GitHub Actions. Model metrics and confusion matrix outputs are
generated during each CI run.
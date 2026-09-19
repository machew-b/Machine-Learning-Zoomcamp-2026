# Project setup

This repository is organized as a learning portfolio for the Machine Learning Zoomcamp course. The environment below is intentionally simple and reproducible.

## 1) Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Start Jupyter

```bash
jupyter notebook
```

## 4) Keep notebook execution consistent

- Prefer the pinned versions in `requirements.txt`.
- If a notebook depends on a specific library version, note it in that notebook's README.
- Run the project from the repository root to keep relative paths stable.

## 5) Project conventions

- Use `../datasets/<file>.csv` for shared course datasets.
- Place each homework in its own folder with a `README.md` and notebook.
- Keep model artifacts and generated outputs out of source control via `.gitignore`.

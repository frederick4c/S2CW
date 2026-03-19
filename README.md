# S2 Coursework: Coal Mining Accident Analysis
Fred Lawrence -fl482

## Repository Structure

| File | Description |
|------|-------------|
| `notebook.ipynb` | All code for exercises 1–5 (data loading, MCMC, nested sampling, plotting) |
| `RJMCMC.py` | Reversible jump MCMC class (Green 1995), completed as part of exercise 5a |
| `written_answers.tex` | LaTeX source for written answers |
| `written_answers.pdf` | Compiled written answers (compile from `written_answers.tex`) |
| `figs/` | All generated figures, referenced in `written_answers.pdf` |
| `coal_mining_accident_data.dat` | Intervals (days) between accidents |
| `pyproject.toml` | Project metadata and dependencies |

## Setup

```bash
python -m venv S2CWvenv
source S2CWvenv/bin/activate
pip install -e .
```

Then launch the notebook:

```bash
jupyter notebook notebook.ipynb
```

## Compiling the Written Answers

```bash
pdflatex written_answers.tex
```
# Tennis ML Lab

Tennis ML Lab is a Streamlit application for experimenting with ATP match prediction and betting-strategy research. It builds chronological, leakage-aware features from historical matches, trains several scikit-learn models, evaluates them with walk-forward validation, and lets you compare backtest strategies on the same test period.

The application is intended for research and education. Backtest results are not a guarantee of future performance and should not be interpreted as financial advice.

## Features

- Chronological Elo, surface Elo, recent form, head-to-head, ranking, age, height, experience, and market-odds features.
- Logistic regression, decision tree, random forest, gradient boosting, and KNN models.
- Walk-forward validation, calibration metrics, AUC, log loss, Brier score, and accuracy.
- Flat-stake and fractional-Kelly backtesting with ROI, profit, drawdown, bankroll, and surface breakdowns.
- Random model/feature search and ROI-strategy search.
- Model persistence in SQLite, Joblib, and Parquet.
- Upcoming ATP matches with optional odds from The Odds API, odds-api.io, and public Polymarket markets.
- A data-management tab that downloads and consolidates the historical sources from scratch.

## Requirements

- Python 3.11 to 3.13 (Python 3.13 is recommended on Windows)
- Git, available on `PATH` (needed to download the historical repositories)
- Internet access for the first data build and optional live odds providers

The project does not require a pre-existing database, model, cache, or local data folder. Those files are generated at runtime and are intentionally excluded from Git.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/AStuderTh/ml-app.git
cd ml-app
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the app with the virtual-environment executable directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

If Windows reports `DLL load failed` while importing pandas, repair the virtual environment's native wheels:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir pandas numpy
```

If that does not work, remove `.venv`, recreate it with Python 3.13, and run the installation commands again. Some Windows application-control policies block native SciPy DLLs in Python 3.14 environments.

### macOS/Linux

```bash
git clone https://github.com/AStuderTh/ml-app.git
cd ml-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build the database from scratch

Start the application:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the **Data** tab and click **Update database**. On a fresh checkout, the update process automatically:

1. Creates the `data/` directory.
2. Clones or updates the `tennis_atp` history repository.
3. Clones or updates the `TML-Database` history repository.
4. Downloads the current and previous seasons from `tennis-data.co.uk`.
5. Consolidates the three sources, removes duplicates, and writes `data/tennis.db`.

The same process can be run from a terminal:

```bash
python scripts/update_data.py
```

The generated database is recreated on every consolidation. If an ambiguous match is found, review `data/match_review.csv`, set its `decision` column to `merge` or `reject`, and run the update again.

## Optional odds APIs

The historical model and backtests work without API keys. Live odds integrations are optional. Create `.streamlit/secrets.toml` locally and add only the keys you have:

```toml
ODDS_API_KEY = "your-the-odds-api-key"
ODDS_API_IO_KEY = "your-odds-api-io-key"
```

`.streamlit/secrets.toml` is ignored by Git. Never commit real API keys.

## Windows launcher

After installation, double-click `lancer_app.bat`. It uses `.venv` automatically when it exists and otherwise falls back to the Python executable available on `PATH`.

## Project layout

```text
app/                    Streamlit UI and ML/backtesting modules
scripts/update_data.py  Download and consolidation entry point
scripts/consolidate/    Source loaders, matching, review, and SQLite writer
requirements.txt        Python dependencies
lancer_app.bat          Windows launcher
data/                   Generated database, caches, models, and source data
```

## Data sources

- [tennis_atp](https://github.com/Kadantte/tennis_atp)
- [TML-Database](https://github.com/Tennismylife/TML-Database)
- [tennis-data.co.uk](http://www.tennis-data.co.uk/)
- Optional live odds: [The Odds API](https://the-odds-api.com/), [odds-api.io](https://odds-api.io/), and [Polymarket](https://polymarket.com/)

Please check each provider's terms and license before redistributing downloaded data.
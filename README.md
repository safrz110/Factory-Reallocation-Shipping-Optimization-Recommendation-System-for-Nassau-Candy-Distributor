# 🍬 Nassau Candy Distributor — Factory Optimization Dashboard

**A machine learning system that predicts shipping lead times and recommends optimal factory-to-region assignments**, built with scikit-learn and deployed as an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

[Live Demo](https://factory-reallocation-shipping-optimization-recommendation-0.streamlit.app) · [Notebook Walkthrough](#notebook) · [Report a Bug](#)

---

## Overview

Nassau Candy Distributor ships products from five regional factories out to four U.S. delivery regions. Every product-factory-region-shipping combination has a different lead time, and the "obvious" factory assignment isn't always the fastest or most profitable one.

This project builds a **Gradient Boosting model that predicts delivery lead time** from product, factory, region, shipping mode, and order-level features, then wraps that model in a dashboard that:

- Simulates lead time across every factory for a given product and region
- Recommends factory reassignments that cut lead time, ranked by expected impact and confidence
- Lets planners run what-if scenarios across regions and shipping modes
- Surfaces risk classification and profit impact for each product

It was built end-to-end — from data cleaning and feature engineering through model benchmarking, clustering, and a production-style Streamlit front end.

## Demo

> (https://factory-reallocation-shipping-optimization-recommendation-0.streamlit.app)
 

## Key Features

- **Lead-time prediction engine** — a Gradient Boosting Regressor trained on shipping distance, ship mode, product economics, and route interaction features
- **Factory Simulator tab** — compares predicted lead time across all five factories for any product/region/ship-mode combination, plus a geographic map of factories and regions
- **What-If Scenario tab** — current-vs-recommended lead time across every region and ship mode, plus a factory × region improvement heatmap
- **Ranked Recommendations tab** — every product's best alternative factory, ranked by improvement %, days saved, and a confidence score
- **Risk & Impact tab** — profit-uplift estimates per product, a high/medium/low risk classification based on lead time and profitability, model diagnostics (actual vs. predicted, feature importance)
- **Speed ↔ Profit priority slider** — a single control that re-weights every recommendation along the speed/profit trade-off
- **Cached model training** — the model trains once per session via Streamlit's resource cache, keeping the app responsive

## How It Works

**1. Feature engineering** — Factory and region coordinates are used to compute haversine shipping distance; ship mode maps to a base transit time; product, region, division, factory, and ship mode are label-encoded; interaction features (distance × ship days, ship mode × division) and economics features (profit margin, unit value) round out the feature set.

**2. Outlier removal** — Lead times outside the 1st–99th percentile are trimmed before training.

**3. Model benchmarking** — Three models are trained and compared with 5-fold cross-validation in the development notebook:

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Ensemble benchmark |
| **Gradient Boosting** | **Best model — used in production** |

Each is scored on **R²**, **RMSE**, and **MAE** on a held-out test split; Gradient Boosting (`n_estimators=100`, `max_depth=5`, `learning_rate=0.10`, `subsample=0.85`) came out on top and is the model that powers the live dashboard.

**4. Route clustering & scenario simulation** — Routes and products are clustered to flag consistently slow lanes, and a simulation engine sweeps every product × factory × region × ship-mode combination to surface reassignment opportunities.

**5. Optimization & recommendations** — For each product, the alternative factory with the largest average lead-time improvement (across regions and ship modes) is selected, paired with a confidence score that rewards consistency across scenarios, not just average improvement.

**6. KPI dashboard** — Aggregates total orders, revenue, profit margin, average lead time, best model R², and top improvement opportunities into a single planning summary.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (Linear Regression, Random Forest, Gradient Boosting), joblib |
| Visualization | Plotly (Express & Graph Objects), Matplotlib (notebook) |
| App / UI | Streamlit, custom CSS theming |

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn
```

> Prefer a single command? Generate a `requirements.txt` from the imports above and run `pip install -r requirements.txt`.

### Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (typically `http://localhost:8501`).

> **Note on data:** the development notebook trains and evaluates the models against the original order-level dataset (`data.csv`). The standalone Streamlit app reconstructs a representative dataset with matching feature distributions and relationships at runtime, so the dashboard runs fully self-contained without requiring the original file — the modeling pipeline, features, and Gradient Boosting model are identical to what's developed in the notebook.

## Project Structure

```
.
├── app.py                              # Streamlit dashboard (model training, simulation, UI)
├── Nassau_Candy_Optimization.ipynb     # Data cleaning, EDA, model development & evaluation notebook
├── data.csv                            # Source order-level data (not included — add your own)
└── README.md
```

## Notebook

The accompanying Jupyter notebook (`Nassau_Candy_Optimization.ipynb`) documents the full analytical process behind the dashboard, step by step: dataset loading and inspection, exploratory data analysis, feature engineering, outlier removal, the train/test split, predictive modeling (Linear Regression, Random Forest, Gradient Boosting) with cross-validation, model evaluation and comparison, feature importance analysis, route and product clustering, the scenario simulation engine, optimization and recommendation logic, KPI computation, and saving the final model with joblib. It's a good companion read if you want to see the reasoning behind each modeling decision before diving into the dashboard code.

## Roadmap

- [ ] Connect the app to a live or refreshed order dataset instead of a synthetic stand-in
- [ ] Add hyperparameter tuning (grid/Bayesian search) for the Gradient Boosting model
- [ ] Persist the trained model with `joblib` and load it at app startup to skip retraining
- [ ] Add automated tests for the lead-time prediction and recommendation logic
- [ ] Deploy to Streamlit Community Cloud and link the live demo above
- [ ] Extend optimization to jointly consider multi-product factory capacity constraints

## Author

**Sarfaraz Ali**


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<sub>Built as a portfolio project demonstrating end-to-end ML pipeline development, from data cleaning through model deployment.</sub>

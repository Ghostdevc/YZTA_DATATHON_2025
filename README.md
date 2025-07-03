✅ YZTA_DATATHON_2025

# YZTA Datathon 2025 – Price Forecasting Project

This repository contains a solution developed for the YZTA 2025 Datathon. The task is to forecast future product prices using historical data. The approach involves time-based feature engineering and training a LightGBM model for regression.

## Project Overview

- Forecasts future prices for various product-market-city combinations.
- Utilizes lag features, rolling statistics, and date-based encodings.
- Trains a LightGBM model for time series regression.
- Provides prediction outputs in the required datathon submission format.

## Tech Stack

- Python
- Pandas, NumPy
- LightGBM
- Matplotlib, Seaborn

## How to Run

1. Clone the repository:
git clone https://github.com/Ghostdevc/YZTA_DATATHON_2025.git
cd YZTA_DATATHON_2025

2. Install required packages:
pip install -r requirements.txt

3. Run the pipeline:
python main.py

The script will:
- Load and preprocess the dataset.
- Generate lag and rolling window features.
- Train and validate a LightGBM model.
- Save final predictions to a `.csv` file.

## Dataset Split

- Train: 2019-01-01 to 2022-12-01  
- Validation: 2023-01-01 to 2023-12-01  
- Test: 2024-01-01 to 2024-12-01

## Outputs

- Model evaluation metrics on the validation set
- Feature importance chart
- Submission-ready prediction file

## License

MIT License

# Bitcoin Volatility Modeling (ARCH, GARCH, EGARCH)

This project analyzes and forecasts the volatility of Bitcoin returns using ARCH-family models.


## Project Structure

- data_raw/        # Raw Bitcoin price data (ignored by Git)
- data_clean/      # Processed datasets
- notebooks/       # Exploratory analysis (EDA, plots)
- src/             # Model estimation scripts
- output/          # Figures and results
- thesis/          # LaTeX thesis files


## Data

Daily Bitcoin price data is used.

Log returns are computed as:
r_t = log(P_t) - log(P_{t-1})


## Setup

1. Create a virtual environment:
   python -m venv venv

2. Activate environment:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt


## Usage

1. Place raw data in `data_raw/`
2. Run notebooks in `notebooks/` for data exploration
3. Run scripts in `src/` for model estimation
4. Outputs are saved in `output/`


## Models

- ARCH
- GARCH
- EGARCH



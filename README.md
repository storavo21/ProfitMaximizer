# Trading Data Analyzer

A focused tool to analyze trading data and identify specific column-based filters that maximize profit potential, specifically targeting 10x+ returns.

## Features

- Load and analyze Excel trading sheets (.xlsx format)
- Identify specific filters based on column values that maximize profit potential
- Calculate which combinations of column values lead to the highest win rates
- Generate a clear report of recommended filters to apply
- Provide a simple interface to view and understand the results
- Allow basic sorting and exploration of the most profitable patterns

## Usage

1. Upload your Excel trading data file using the file uploader
2. The app will automatically analyze your data and provide summary metrics
3. Select columns to analyze for profitable patterns
4. Adjust the minimum trades and target return parameters as needed
5. View and sort the identified profitable filters
6. Visualize filter effectiveness with interactive charts
7. Download the analysis results as CSV or Excel file

## Data Format

Your Excel file should contain columns with information about your trades. At minimum, you need:

- A column showing profit/return/result for each trade
- Additional columns with factors you want to analyze (e.g., entry time, strategy, market conditions)

Example columns that work well with this tool:

- **profit/return/result**: The profit or return percentage for each trade
- **entry_time**: Time when you entered the trade
- **symbol**: The trading symbol or instrument
- **strategy**: Trading strategy used
- **market_condition**: Bull market, bear market, etc.
- **position_size**: Size of the position
- **trade_duration**: How long the trade was active

## Running the Application

To run the application locally:

```bash
streamlit run app.py --server.port 5000

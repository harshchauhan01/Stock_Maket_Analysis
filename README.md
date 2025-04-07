
# Stock Market Comparision Analysis

This project focuses on analyzing and predicting stock market trends using real-time data. Stock prices are fetched directly using the yfinance Python module, which allows access to historical and live data for multiple companies.

The key features include:

Real-time data fetching via yfinance

Visualization of daily, monthly, and yearly stock trends

Machine Learning-based prediction using Linear Regression, LSTM, and RNN models

Accuracy comparison revealing that Linear Regression outperformed LSTM and RNN on this dataset

A clean and interactive GUI-based dashboard to explore stock trends across companies

This project integrates data science, machine learning, finance, and Python GUI development into a comprehensive stock analysis tool.


## Run Locally

Clone the project

```bash
  git clone https://github.com/harshchauhan01/Stock_Maket_Analysis.git
```

Go to the project directory

```bash
  cd <my-project-directory>
```

Install dependencies

```bash
  python -m venv venv
  venv\Scripts\activate
  pip install yfinace
  pip install sklearn
  pip install tensorflow
  pip install matplotlib, numpy, pandas, seaborn
```

Start the server

```bash
  streamlit run stock_analysis.py
```


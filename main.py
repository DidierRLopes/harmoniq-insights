import json
from pathlib import Path
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import plotly.graph_objects as go
from plotly_config import create_base_layout, apply_config_to_figure
from registry import WIDGETS, register_widget
from datetime import datetime, timedelta
import numpy as np
from openbb import obb


app = FastAPI()

origins = [
    "https://pro.openbb.co",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_PATH = Path(__file__).parent.resolve()

@app.get("/")
def read_root():
    return {"Info": "HarmoniQ Insights App"}


@app.get("/widgets.json")
async def get_widgets():
    return WIDGETS

@app.get("/templates.json")
async def get_templates():
    with open(ROOT_PATH / "templates.json", "r") as f:
        return json.load(f)
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/equities_table")
@register_widget({
    "name": "Equities Overview",
    "description": "Shows performance metrics for major global equity indices",
    "category": "Equities",
    "type": "table",
    "endpoint": "equities_table",
    "gridData": {"w": 80, "h": 9},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "Index",
                "field": "Index",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    }
})
def get_equities_table():
    """Get performance metrics for major global equity indices"""
    symbols = {
        "^NDX": "NASDAQ",
        "^GSPC": "S&P 500",
        "VEU": "IEF (Global Ex-US)",
        "FEZ": "STOXX 50",
        "^N225": "NIKKEI",
        "ASHR": "CSI 300",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'Index': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results


@app.get("/bonds_table")
@register_widget({
    "name": "Bonds Overview",
    "description": "Shows performance metrics for major US Treasury and Corporate Bond ETFs",
    "category": "Fixed Income",
    "type": "table",
    "endpoint": "bonds_table",
    "gridData": {"w": 80, "h": 8},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "ETF",
                "field": "ETF",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    },
})
def get_bonds_table():
    """Get performance metrics for major US Treasury and Corporate Bond ETFs"""
    symbols = {
        "SHY": "US Treasury ETF (1-3Y)",
        "IEF": "US Treasury ETF (7-10Y)",
        "TLT": "US Treasury ETF (20Y+)",
        "HYG": "US HY ETF",
        "LQD": "US Inv Grade ETF",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'ETF': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results


@app.get("/commodities_table")
@register_widget({
    "name": "Commodities Overview",
    "description": "Shows performance metrics for major commodity ETFs",
    "category": "Commodities",
    "type": "table",
    "endpoint": "commodities_table",
    "gridData": {"w": 80, "h": 9},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "ETF",
                "field": "ETF",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    },
})
def get_commodities_table():
    """Get performance metrics for major commodity ETFs"""
    symbols = {
        "GLD": "Gold",
        "SLV": "Silver",
        "CPER": "Copper",
        "URA": "Uranium",
        "BNO": "Brent Crude Oil",
        "USO": "WTI Crude Oil",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'ETF': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results


@app.get("/currencies_table")
@register_widget({
    "name": "Currencies Overview",
    "description": "Shows performance metrics for major currency pairs",
    "category": "Currencies",
    "type": "table",
    "endpoint": "currencies_table",
    "gridData": {"w": 80, "h": 8},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "Currency",
                "field": "Currency",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    },
})
def get_currencies_table():
    """Get performance metrics for major currency pairs"""
    symbols = {
        "DX-Y.NYB": "US Dollar Index (DXY)",
        "EURUSD=X": "EUR/USD",
        "JPYUSD=X": "JPY/USD",
        "GBPUSD=X": "GBP/USD",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'Currency': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results


@app.get("/volatility_table")
@register_widget({
    "name": "Volatility Indices Overview",
    "description": "Shows performance metrics for major volatility indices",
    "category": "Volatility",
    "type": "table",
    "endpoint": "volatility_table",
    "gridData": {"w": 80, "h": 6},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "Index",
                "field": "Index",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    },
})
def get_volatility_table():
    """Get performance metrics for major volatility indices"""
    symbols = {
        "^VIX": "VIX",
        "MOVE": "MOVE",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'Index': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results


@app.get("/digital_assets_table")
@register_widget({
    "name": "Digital Assets Overview",
    "description": "Shows performance metrics for major digital assets",
    "category": "Digital Assets",
    "type": "table",
    "endpoint": "digital_assets_table",
    "gridData": {"w": 80, "h": 3},
    "source": "Yahoo Finance",
    "params": [],
    "data": {
      "table": {
        "showAll": True,
        "columnsDefs": [
            {
                "headerName": "Asset",
                "field": "Asset",
                "minWidth": 200
            },
            {
                "headerName": "5D",
                "field": "5D",
                "minWidth": 150
            },
            {
                "headerName": "MTD",
                "field": "MTD",
                "minWidth": 150
            },
            {
                "headerName": "YTD",
                "field": "YTD",
                "minWidth": 150
            },
            {
                "headerName": "5Y (CAGR)",
                "field": "5Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "10Y (CAGR)",
                "field": "10Y (CAGR)",
                "minWidth": 150
            },
            {
                "headerName": "Value",
                "field": "Value",
                "minWidth": 200
            }
        ]
      }
    },
})
def get_digital_assets_table():
    """Get performance metrics for major digital assets"""
    symbols = {
        "BTC-USD": "Bitcoin USD",
        "ETH-USD": "Ethereum USD",
        "SOL-USD": "Solana USD",
    }

    # Get dates for different time periods
    end_date = datetime.now()
    start_ytd = datetime(end_date.year, 1, 1)
    start_mtd = datetime(end_date.year, end_date.month, 1)
    start_5d = end_date - timedelta(days=5)
    start_5y = datetime(end_date.year - 5, end_date.month, end_date.day)
    start_10y = datetime(end_date.year - 10, end_date.month, end_date.day)

    results = []

    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_10y.strftime("%Y-%m-%d")
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            latest_price = round(df_historical['close'].iloc[-1], 2)
            
            # Calculate 5-day change
            df_historical.index = pd.to_datetime(df_historical.index)
            five_day_mask = df_historical.index >= pd.to_datetime(start_5d)
            if five_day_mask.any():
                five_day_idx = df_historical.index[five_day_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                five_day_close = df_historical.loc[five_day_idx, 'close']
                five_day_change_pct = (
                    (latest_close - five_day_close) / five_day_close
                ) * 100
            else:
                five_day_change_pct = np.nan
            
            # Calculate MTD change
            mtd_mask = df_historical.index >= pd.to_datetime(start_mtd)
            if mtd_mask.any():
                mtd_idx = df_historical.index[mtd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                mtd_close = df_historical.loc[mtd_idx, 'close']
                mtd_change_pct = (
                    (latest_close - mtd_close) / mtd_close
                ) * 100
            else:
                mtd_change_pct = np.nan
            
            # Calculate YTD change
            ytd_mask = df_historical.index >= pd.to_datetime(start_ytd)
            if ytd_mask.any():
                ytd_idx = df_historical.index[ytd_mask][0]
                latest_close = df_historical['close'].iloc[-1]
                ytd_close = df_historical.loc[ytd_idx, 'close']
                ytd_change_pct = (
                    (latest_close - ytd_close) / ytd_close
                ) * 100
            else:
                ytd_change_pct = np.nan
            
            # Calculate 5Y CAGR
            five_y_mask = df_historical.index >= pd.to_datetime(start_5y)
            if five_y_mask.any():
                five_y_data = df_historical[five_y_mask]
                start_5y_price = five_y_data.iloc[0]['close']
                years_5 = (end_date - start_5y).days / 365.25
                five_y_cagr = (((latest_price / start_5y_price) ** (1 / years_5)) - 1) * 100
            else:
                five_y_cagr = np.nan
            
            # Calculate 10Y CAGR
            ten_y_mask = df_historical.index >= pd.to_datetime(start_10y)
            if ten_y_mask.any():
                ten_y_data = df_historical[ten_y_mask]
                start_10y_price = ten_y_data.iloc[0]['close']
                years_10 = (end_date - start_10y).days / 365.25
                ten_y_cagr = (((latest_price / start_10y_price) ** (1 / years_10)) - 1) * 100
            else:
                ten_y_cagr = np.nan
            
            results.append({
                'Asset': name,
                '5D': round(float(five_day_change_pct), 1) if not np.isnan(five_day_change_pct) else None,
                'MTD': round(float(mtd_change_pct), 1) if not np.isnan(mtd_change_pct) else None,
                'YTD': round(float(ytd_change_pct), 1) if not np.isnan(ytd_change_pct) else None,
                '5Y (CAGR)': round(float(five_y_cagr), 1) if not np.isnan(five_y_cagr) else None,
                '10Y (CAGR)': round(float(ten_y_cagr), 1) if not np.isnan(ten_y_cagr) else None,
                'Value': round(float(latest_price), 1)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    return results

@app.get("/equity_performance")
@register_widget({
    "name": "Equity Performance",
    "description": "Shows normalized performance of major equity indices",
    "category": "Equities",
    "type": "chart",
    "endpoint": "equity_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_equity_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major equity indices"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major equity indices
    symbols = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng"
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())

@app.get("/bonds_performance")
@register_widget({
    "name": "Bonds Performance",
    "description": "Shows normalized performance of major bond ETFs",
    "category": "Fixed Income",
    "type": "chart",
    "endpoint": "bonds_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_bonds_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major bond ETFs"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major bond ETFs
    symbols = {
        "SHY": "US Treasury ETF (1-3Y)",
        "IEF": "US Treasury ETF (7-10Y)",
        "TLT": "US Treasury ETF (20Y+)",
        "HYG": "US HY ETF",
        "LQD": "US Inv Grade ETF"
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())

@app.get("/commodities_performance")
@register_widget({
    "name": "Commodities Performance",
    "description": "Shows normalized performance of major commodity ETFs",
    "category": "Commodities",
    "type": "chart",
    "endpoint": "commodities_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_commodities_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major commodity ETFs"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major commodity ETFs
    symbols = {
        "GLD": "Gold",
        "SLV": "Silver",
        "CPER": "Copper",
        "URA": "Uranium",
        "BNO": "Brent Crude Oil",
        "USO": "WTI Crude Oil"
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())

@app.get("/currencies_performance")
@register_widget({
    "name": "Currencies Performance",
    "description": "Shows normalized performance of major currency pairs",
    "category": "Currencies",
    "type": "chart",
    "endpoint": "currencies_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_currencies_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major currency pairs"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major currency pairs
    symbols = {
        "DX-Y.NYB": "US Dollar Index (DXY)",
        "EURUSD=X": "EUR/USD",
        "JPYUSD=X": "JPY/USD",
        "GBPUSD=X": "GBP/USD"
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())

@app.get("/volatility_performance")
@register_widget({
    "name": "Volatility Performance",
    "description": "Shows normalized performance of major volatility indices",
    "category": "Volatility",
    "type": "chart",
    "endpoint": "volatility_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_volatility_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major volatility indices"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major volatility indices
    symbols = {
        "^VIX": "VIX",
        "MOVE": "MOVE"
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())

@app.get("/digital_assets_performance")
@register_widget({
    "name": "Digital Assets Performance",
    "description": "Shows normalized performance of major digital assets",
    "category": "Digital Assets",
    "type": "chart",
    "endpoint": "digital_assets_performance",
    "gridData": {"w": 40, "h": 15},
    "source": "Yahoo Finance",
    "data": {"chart": {"type": "line"}},
    "params": [
        {
            "paramName": "start_date",
            "value": "ytd",
            "label": "Start Date",
            "show": True,
            "description": "Starting date for performance comparison (ytd, mtd, 5d, 5y, 10y)",
            "type": "text",
            "options": [
                {"label": "Year to Date", "value": "ytd"},
                {"label": "Month to Date", "value": "mtd"},
                {"label": "5 Days", "value": "5d"},
                {"label": "5 Years", "value": "5y"},
                {"label": "10 Years", "value": "10y"}
            ]
        }
    ],
})
def get_digital_assets_performance(start_date: str = "ytd", theme: str = "dark"):
    """Get normalized performance of major digital assets"""
    # Define time periods
    end_date = datetime.now()
    time_periods = {
        '5d': end_date - timedelta(days=5),
        'mtd': datetime(end_date.year, end_date.month, 1),
        'ytd': datetime(end_date.year, 1, 1),
        '5y': datetime(end_date.year - 5, end_date.month, end_date.day),
        '10y': datetime(end_date.year - 10, end_date.month, end_date.day)
    }
    
    # Get start date based on parameter
    start_date = time_periods.get(start_date.lower(), time_periods['ytd'])
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Major digital assets
    symbols = {
        "BTC-USD": "Bitcoin USD",
        "ETH-USD": "Ethereum USD",
        "SOL-USD": "Solana USD",
    }

    # Create a plotly figure with base layout
    figure = go.Figure(
        layout=create_base_layout(
            x_title="Date",
            y_title="Normalized Price (Base=100)",
            theme=theme
        )
    )

    # Fetch and plot data for each symbol
    for symbol, name in symbols.items():
        try:
            # Fetch historical data using OpenBB
            df_historical = obb.equity.price.historical(
                symbol=symbol,
                provider="yfinance",
                start_date=start_date_str
            ).to_df()
            
            if df_historical.empty or len(df_historical) < 2:
                continue
            
            # Normalize the data to start at 100 for better comparison
            normalized_data = df_historical['close'] / df_historical['close'].iloc[0] * 100
            
            # Add the data to the plot
            figure.add_trace(go.Scatter(
                x=df_historical.index,
                y=normalized_data,
                mode='lines',
                name=name
            ))
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Apply theme configuration
    figure = apply_config_to_figure(figure, theme)

    # Return the plotly json
    return json.loads(figure.to_json())
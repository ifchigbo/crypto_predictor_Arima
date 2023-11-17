import pandas as pd
from pandas.errors import EmptyDataError, DuplicateLabelError, DtypeWarning
import numpy as np
import json
import requests
from requests import sessions
from requests_html import HTMLSession
import yfinance as yf
from datetime import date


def getCryptCoinSymbols():
    # Yahoo finance web scraping method - trim coins to get 20 coins with WBTC
    try:
        session = HTMLSession()
        num_currencies = 100 #pull a list of 100 coins
        url = f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}"
        resp = session.get(url)
        tables = pd.read_html(resp.html.raw_html)
        #print(tables)
        df = tables[0].copy()
        symbols_yf = df.Symbol.tolist()
        coins_of_interest = symbols_yf[:19] + symbols_yf[20:21]#get a list of  20 coins including WBTC
        cryp_symbol = []
        for symbols in coins_of_interest:#symbols_yf[:20]:  # I am using this list to save a copy of 20 coins
            cryp_symbol.append(symbols)
        return cryp_symbol
    except BaseException as error:
        print(error)



#Pick Pandas dataframe per  coin for the list above in crypt-coin symbols
def listAllCoins(cfdata):
    start_date = '2022-01-01'
    all_coins = getCryptCoinSymbols()
    end_date = date.today().strftime('%Y-%m-%d')
    try:
        list_coins = []
        for items in all_coins:
            data = yf.Ticker(items)
            datasets = data.history(start=start_date, end=end_date)
            datasets['logo'] = items
            list_coins.append(datasets)
        df = pd.concat(list_coins)

        # use the condition below to drop nulll values if they exist any time in future - the code pulls data real time
        if df.isna().sum().sum().astype(int) != 0:

            df.dropna(axis=0, inplace=True)

        # end of the drop null value logis

        return df.loc[df['logo'] == cfdata]

    except BaseException as error:
        print(error)

#print(getCryptCoinSymbols()) # Debugger to test output of web scrapping


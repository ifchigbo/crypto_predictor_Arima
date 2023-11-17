from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import plotly.express as px
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
import pandas as pd
import numpy as np
import os, pickle, time, threading, json
from getCoins import getCryptCoinSymbols, listAllCoins
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dash.exceptions import PreventUpdate
from datetime import datetime

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], 
           title='Coin Predictor')

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60  #Cache Coins for 60 secs

@cache.memoize(timeout=TIMEOUT)
def get_coins():
    try:
        coin_list = getCryptCoinSymbols()
        return json.dumps(coin_list)
    except BaseException as error:
        print(error)

def list_coins():
    try:
        return json.loads(get_coins())
    except  BaseException as error:
        print(error)

# coin_list = getCoins()

coin_list_dropdown = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(list_coins(), id='coin-selector', placeholder='Select a Crypto Coin'),
        )
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    style={'width': '60%'}
)
app.layout = html.Div([
    dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("SoliGence Crypto Coin Predictor", class_name='ms-2'),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    coin_list_dropdown,
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ]
        ),
        color='dark',
        dark=True
    ),
    dbc.Container([
        html.Div([
            html.Div([
                html.P("Low"),
                html.Div([
                    dcc.Slider(min=2, max=30, step=2, value=2, id='moving-average-low')
                ]),
                html.P("High"),
                html.Div([
                    dcc.Slider(min=2, max=30, step=2, value=4, id='moving-average-high')
                ]),
            ], style={'backgroundColor': 'white', 'padding': '2em'}),
            dcc.Loading(type="default", children=dcc.Graph(id='moving-average-trend', style={'z-index': -1})),
        ], style={'padding': '4em', 'backgroundColor': 'whitesmoke'}),
        
        html.Div([
            html.Div([
                #html.H5('Select Forecast duration from slider', style={'padding': '1em 0'}),
                html.H5('Select Prediction Day Range', style={'padding': '1em 0'}),
                #dcc.Slider(min=0, max=30, step=1, value=1, id='observation-value'), # test  sliders
                dcc.Slider(min=0, max=30, step=1, value=1, id='observation-value'),
                dcc.Loading(type="default", children=html.Div(id='ar-model-forecast', style={'padding': '1em 0'})),
                dcc.Loading(type="default", children=dcc.Graph(id='ar-model-trend')),
                dcc.Loading(type="default", children=html.Div(id='model-statistics')),
            ]),
        ]),
        #dcc.Graph(id='price-trend'),
        html.H2("Profit Margin Calculator", style={'margin' : '2em 0'}),
        html.Div([
            html.Div([
                html.Label(htmlFor='coin-quantity', children=html.Span('Coin Quantity to Buy')),
                dcc.Input(
                    id='coin-quantity',
                    type='number',
                    className='form-control',
                    style={'width': '60%'},
                    value = 1,
                    min = 1
                )
            ], className='form-group mb-3 m-auto'),
            html.Div([
                html.Label(htmlFor='num-of-days', children=html.Span('Days of Planned Investment')),
                dcc.Input(
                    id='num-of-days',
                    type='number',
                    className='form-control',
                    style={'width': '60%'},
                    value = 1,
                    min=1
                )
            ], className='form-group mb-3 m-auto'),
            dcc.Loading(html.Div(id='price-list')),
        ], style={'minHeight': '40vh'}),

    ]),
])

#@app.callback(
    #Output('price-trend', 'figure'),
    #Input('coin-selector', 'value')
#)
# def update_price_graph(coin_value):
#
#     if coin_value is None:
#         raise PreventUpdate()
#     data = listAllCoins(coin_value)
#     fig = px.line(data, x=data.index, y='Open', title=f'{coin_value} Stock Plot', markers=True)
#     fig.update_traces(name='Open', showlegend=True)
#     fig.add_scatter(x=data.index, y=data['Close'], mode='lines', name='Close')
#     fig.update_layout(yaxis_title=f'{coin_value} value in USD ($)')
#     return fig

def getSimpleMovingAverage(cryptdata, period=30, column='Close'):
    try:
        return cryptdata[column].rolling(window=period).mean()
    except BaseException as error:
        print(error)

@app.callback(
    Output('moving-average-trend', 'figure'),
    Input('moving-average-low', 'value'),
    Input('moving-average-high', 'value'),
    Input('coin-selector', 'value'),
)
def update_moving_average_graph(low, high, coin_value):

    if coin_value is None:
        raise PreventUpdate()
    sma_data = listAllCoins(coin_value)
    sma_data[f'MA{low}']=getSimpleMovingAverage(sma_data, low)
    sma_data[f'MA{high}']=getSimpleMovingAverage(sma_data, high)
    sma_data['trigger'] = np.where(sma_data[f'MA{low}'] > sma_data[f'MA{high}'],1,0)
    sma_data['position'] = sma_data['trigger'].diff()
    sma_data['buy'] = np.where(sma_data['position'] == 1, sma_data['Close'],np.NAN)
    sma_data['sell'] = np.where(sma_data['position'] == -1, sma_data['Close'],np.NAN)
    fig = px.line(sma_data, x=sma_data.index, y='Close', height=500, title=f'{coin_value} Moving Average')
    fig.update_traces(name='Close', showlegend=True)
    fig.add_scatter(x=sma_data.index, y=sma_data[f'MA{low}'], mode='lines', name=f'MA{low}')
    fig.add_scatter(x=sma_data.index, y=sma_data[f'MA{high}'], mode='lines', name=f'MA{high}')
    fig.add_scatter(x=sma_data.index, y=sma_data['buy'], name='buy', mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'))
    fig.add_scatter(x=sma_data.index, y=sma_data['sell'], name='sell', mode='markers', marker=dict(color='red', size=8, symbol='triangle-up'))
    return fig


@app.callback(
    Output('ar-model-forecast', 'children'),
    Output('ar-model-trend', 'figure'),
    Output('model-statistics', 'children'),
    Input('coin-selector', 'value'),
    Input('observation-value', 'value'),
)
def update_armodel_forecast_table(coin_value, days):
    if coin_value is None:
        raise PreventUpdate()
    c_data = listAllCoins(coin_value)
    rows = int(len(c_data) * 0.7)
    train_data1 = list(c_data[0:rows]['Close'])
    test_data1 = list(c_data[rows:]['Close'])
    predictions = []
    observations = len(test_data1)
    model = None

    coinfilename=f'{coin_value}_model.pkl' ## Handling Pickle File
    if os.path.exists(coinfilename):
        with open(coinfilename,'rb') as f:
            model = pickle.load(f)

        forecastseries = model.forecast(int(days) + 1) # Prediction - Seven Days Prediction

        last_date = c_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=len(forecastseries),freq='D')
        forecasts = {'Date': forecast_dates[1:].date, f'{coin_value} Predicted Value': forecastseries[1:]}
        ar_model_forecast = pd.DataFrame(forecasts)
        ar_table = dash_table.DataTable(data=ar_model_forecast.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in ar_model_forecast.columns],
                                        )
        fig = px.line(ar_model_forecast, x='Date', y=f'{coin_value} Predicted Value', title=f'{coin_value} Prediction Trend')

        return (ar_table, fig, [])
    else:
        AR_model = None
        test_data = c_data.iloc[rows:]
        for element in np.arange(observations):
            model = ARIMA(train_data1, order=(3, 1, 2))
            AR_model = model.fit()
            element_prediction = AR_model.forecast(method_kwargs={"warn_convergence": False})
            predictions.append(element_prediction[0])
            actual_test_value = test_data1[element]
            train_data1.append(actual_test_value)

        with open(coinfilename, 'wb') as f: ## Handling Pickle File
            pickle.dump(AR_model,f)    ## Handling Pickle File
        date_range = list(c_data[rows:].index)
        mape = mean_absolute_percentage_error(test_data1,predictions)
        mae = mean_absolute_error(test_data1, predictions)
        test_data['Predicted Value'] = predictions
        last_date = c_data.index[-1]
        forecastseries = AR_model.forecast(days) # Prediction - Seven Days Prediction
        forecast_dates = pd.date_range(start=last_date, periods=len(forecastseries),freq='D')
        foreecasts = {'Date': forecast_dates[1:].date, f'{coin_value} Predicted Value': forecastseries[1:]}
        ar_model_forecast = pd.DataFrame(foreecasts)
        ar_table = dash_table.DataTable(data=ar_model_forecast.to_dict('records'),
                                        columns=[{'name': i, 'id': i} for i in ar_model_forecast.columns])
        fig = px.line(x=date_range, y=predictions, title=f'{coin_value} Price Prediction Evaluation and Metrics')
        fig.update_traces(name=f'{coin_value} predicted value', showlegend=True)
        fig.add_scatter(x=date_range, y=test_data1, mode='lines', name=f'{coin_value} actual value')
        fig.update_layout(xaxis_title='Date', yaxis_title='Values')
        statistics = {'MAPE': [mape], 'MAE': [mae]}
        model_stats = pd.DataFrame(statistics)
        model_stats_table = dash_table.DataTable(data=model_stats.to_dict('records'),
                                                 columns=[{'name': i, 'id': i} for i in model_stats.columns],
                                                 )
        return (ar_table, fig, model_stats_table,)
    
def remove_files_daily():
    for file in os.listdir():
        if file.endswith('.pkl'):
            creation_time = os.path.getctime(file)
            current_time = time.time()
            time_difference = current_time - creation_time
            time_difference_in_hours = time_difference / 3600
            if time_difference_in_hours > 24:
                os.remove(file)

@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('price-list', 'children'),
    Input('ar-model-forecast', 'children'),
    Input('coin-selector', 'value'),
    Input('num-of-days', 'value'),
    Input('coin-quantity', 'value')
)
def update_investment_forecast(active_cell, coin_value, days=1, quantity=1):
    if coin_value is None or days is None or quantity is None:
        raise PreventUpdate()
    todays_date = datetime.now().date()
    #todays_price = None
    #prediction_todays_date = None

    #if active_cell is None: # Update to Active  Cell logical test conditions for  exceptions
    #if not active_cell['props']['data']: # update on test Condition
    #    raise PreventUpdate()
    if active_cell is None:
        raise PreventUpdate()
    if 'props' not in active_cell:
        raise PreventUpdate()
    if 'data' not in active_cell['props']:
        raise PreventUpdate()
    if not active_cell['props']['data']:
        raise PreventUpdate()

    prediction_todays_date = datetime.strptime(active_cell['props']['data'][0]['Date'], '%Y-%m-%d').date()
    print(todays_date)
    print(prediction_todays_date)
    #if todays_date == prediction_todays_date:
    todays_price = active_cell['props']['data'][0][f'{coin_value} Predicted Value']
    if quantity == None:
        quantity = 1
    if days == None:
        days = 1
    coinfilename=f'{coin_value}_model.pkl' ## Handling Pickle File
    if os.path.exists(coinfilename):
        with open(coinfilename,'rb') as f:
            model = pickle.load(f)

        forecastseries = model.forecast(int(days) + 1) # Prediction - Seven Days Prediction
        c_data = listAllCoins(coin_value)
        last_date = c_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=len(forecastseries),freq='D')
        forecasts = {'Date': forecast_dates[1:].date, f'{coin_value} Predicted Value': forecastseries[1:]}
        selling_price = pd.DataFrame(forecasts).tail(1)[f'{coin_value} Predicted Value']
    sp = selling_price.values[0]
    print(sp)
    print(todays_price)

    #Calculation for Profit Margin
    profit = (sp - todays_price) * quantity
    percentage_profit = ((sp - todays_price) / (todays_price * quantity)) * 100
    out = {'Coin': [coin_value], 'Cost Price': [f'{todays_price:.5f}'],
                'Quantity': [quantity], 'Total Price': [f'{(quantity * todays_price):.5f}'],
                f'Selling Price after {days} days': [f'{sp:.5f}'], 'Profit': [f'{profit:.5f}'],
                'Percentage Profit': f'{percentage_profit:.5f}%'}
    out_data = pd.DataFrame(out)
    result_table = dash_table.DataTable(data=out_data.to_dict('records'),
                                                 columns=[{'name': i, 'id': i} for i in out_data.columns],
                                                 )
    return result_table

session_thread = threading.Thread(target=remove_files_daily)
session_thread.start()
    

if __name__ == '__main__':
    app.run_server(debug=True)
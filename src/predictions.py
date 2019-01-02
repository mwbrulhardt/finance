"""
Prints out the predictions for tomorrow for all the stocks entered in the list
'companies'.
"""


import pandas as pd

from src import market as mkt


companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'IBM', 'FB', 'SNAP']

model_1 = {
    'Company': [],
    'Open': [],
    'Close': [],
}

model_2 = {
    'Company': [],
    'Gap': []
}

predictions = {
    'Company': [],
    'Model 1': [],
    'Model 2': []
}

for company in companies:
    stock = mkt.Stock(company, model_path=f'../models/{company}/all_data/')

    # Model 1
    op = stock.predict_next_day('Open')
    cp = stock.predict_next_day('Close')
    d = cp - op
    p1 = 'UP' if d > 0 else 'DOWN'
    model_1['Company'].append(company)
    model_1['Open'].append(round(op, 2))
    model_1['Close'].append(round(cp, 2))

    # Model 2
    gap = stock.predict_next_day('Gap')
    p2 = 'UP' if gap > 0 else 'DOWN'
    model_2['Company'].append(company)
    model_2['Gap'].append(gap)

    # Predictions
    predictions['Company'].append(company)
    predictions['Model 1'].append(p1)
    predictions['Model 2'].append(p2)

model_1_df = pd.DataFrame.from_dict(model_1)
model_2_df = pd.DataFrame.from_dict(model_2)
pred_df = pd.DataFrame.from_dict(predictions)

print('=======================================')
print(model_1_df)
print('=======================================')
print(model_2_df)
print('=======================================')
print(pred_df)
print('=======================================')
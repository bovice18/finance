import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


def fetchStockData(ticker="BTC-USD", beginDate="2018-01-01", endDate="2020-04-30"):
    df = yf.download([ticker],
                     start=beginDate,
                     end=endDate,
                     progress=False)
    return df


def buildLookbackData(dataset, target, start_index, end_index, lookback,
                      target_size):
    data = []
    labels = []

    start_index = start_index + lookback
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-lookback, i)
        data.append(dataset[indices])

        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def buildTensorData(x_train, y_train, x_test, y_test, batch_size, buffer_size=10000):
    # Setup data for RNN model
    # Training
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
    # Testing
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(batch_size).repeat()
    return train_data, test_data


def buildRNN(input_shape, layer1, layer2, output):
    # Model Architecture
    rnn = tf.keras.models.Sequential()
    rnn.add(tf.keras.layers.LSTM(layer1, return_sequences=True, input_shape=input_shape))
    rnn.add(tf.keras.layers.LSTM(layer2, activation='relu'))
    rnn.add(tf.keras.layers.Dense(output))
    rnn.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    return rnn


def plotPredictionTest(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')

    plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')

    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'ro',
                 label='Predicted Future')

    plt.legend(loc='upper left')
    plt.show()

def plotPrediction(model, prices, scaled, start_date, hist_size, predict_size, mean, std, instrument = ['Equity', 'Future', 'Crypto'][0]):
    if instrument == 'Equity':
        nyse = mcal.get_calendar('NYSE').regular_holidays
        holidays = [x.strftime("%Y-%m-%d") for x in nyse.holidays(start='2010-01-01', end='2023-12-31').to_list()]
        beginDate_hist = np.busday_offset(start_date, -hist_size, holidays=holidays)
        endDate_hist = np.busday_offset(start_date, -1, holidays=holidays)

        endDate_future = np.busday_offset(start_date, predict_size - 1, holidays=holidays)
        dates_future = [np.busday_offset(start_date, i, holidays=holidays) for i in range(0, predict_size)]
    elif instrument == 'Future':
        weekdays = "Mon Tue Wed Thu Fri Sun"
        cme = mcal.get_calendar('CME').regular_holidays
        holidays = [x.strftime("%Y-%m-%d") for x in cme.holidays(start='2010-01-01', end='2023-12-31').to_list()]
        beginDate_hist = np.busday_offset(start_date, -hist_size, holidays=holidays, weekmask=weekdays)
        endDate_hist = np.busday_offset(start_date, -1, holidays=holidays, weekmask=weekdays)

        endDate_future = np.busday_offset(start_date, predict_size - 1, holidays=holidays, weekmask=weekdays)
        dates_future = [np.busday_offset(start_date, i, holidays=holidays, weekmask=weekdays) for i in range(0, predict_size)]
    else:
        beginDate_hist = start_date - np.timedelta64(hist_size, 'D')
        endDate_hist = start_date - np.timedelta64(1, 'D')

        endDate_future = start_date + np.timedelta64(predict_size - 1, 'D')
        dates_future = np.arange(start_date, start_date + np.timedelta64(predict_size, 'D'), step=np.timedelta64(1, 'D'))

    # get scaled prices of history
    past_scaled = scaled.loc[beginDate_hist: endDate_hist]
    past_tensor = tf.convert_to_tensor(np.array(past_scaled).reshape(1, past_scaled.shape[0], past_scaled.shape[1]),
                                       dtype=tf.float32)

    # make predictions
    predicted = np.array(model.predict(past_tensor)[0])
    predicted_price = [(x * std) + mean for x in predicted]

    # plot
    trueHistClose = prices.loc[beginDate_hist:endDate_hist][['Close']].rename(columns = {'Close': 'Trailing Price'})
    trueClose = prices.loc[endDate_hist:endDate_future][['Close']].rename(columns = {'Close': 'True Price'})
    predClose = pd.DataFrame.from_dict({'date': dates_future, 'Predicted Price': predicted_price}).set_index('date')
    joined = pd.concat([trueHistClose, trueClose, predClose], axis=1)
    joined.loc[endDate_hist, 'Predicted Price'] = joined.loc[endDate_hist, 'True Price']

    plt.figure()
    joined.plot(figsize=(12,6),
                title='Predictions: {} - {}'.format(dates_future[0], dates_future[-1]),
                )
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.show()

    return joined, predicted_price
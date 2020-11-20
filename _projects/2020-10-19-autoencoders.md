---
title: Outlier Detection with RNN Autoencoders
date: 2020-10-19 22:00:00
description: Article published by Towards Data Science.
featured_image: '/images/demo/outliers.jpg'
---

![](/images/demo/outliers.jpg)

# Outlier Detection with RNN Autoencoders

*Utilising a reconstruction autoencoder model to detect anomalies in time series data.*

TL DR: [Historic-Crypto Package][link1], [Code][link2].

## What are Anomalies?

Anomalies, often referred to as outliers, are data points, data sequences or patterns in data which do not conform to the overarching behaviour of the data series. As such, anomaly detection is the task of detecting data points or sequences which don’t conform to patterns present in the broader data.

The effective detection and removal of anomalous data can provide highly useful insights across a number of business functions, such as detecting broken links embedded within a website, spikes in internet traffic, or dramatic changes in stock prices. Flagging these phenomena as outliers, or enacting a pre-planned response can save businesses both time and money.

## Types of Anomalies?

Anomalous data can typically be separated into three distinct categories, Additive Outliers, Temporal Changes, or Level Shifts.

*Additive Outliers* are characterised by sudden large increases or decreases in value, which can be driven by exogenous or endogenous factors. Examples of additive outliers could be a large increase in website traffic due to an appearance on television (exogenous), or a short-term increase in stock trading volume due to strong quarterly performance (endogenous).

*Temporal Changes* are characterised by a short sequence which doesn’t conform to the broader trend in the data. For example, if a website server crashes, the volume of website traffic will drop to zero for a sequence of datapoints, until the server is rebooted, at which point normal traffic will return.

*Level Shifts* are a common phenomena in commodity markets, as high demand for electricity is inherently linked to inclement weather conditions. As such, a ‘level shift’ can be observed between the price of electricity in summer and winter, owing to weather driven changes in demand profiles and renewable energy generation profiles.

## What is an Autoencoder?

Autoencoders are neural networks designed to learn a low-dimensional representation of a given input. Autoencoders typically consist of two components: an *encoder* which learns to map input data to a lower dimensional representation and a *decoder*, which learns to map the representation back to the input data.

Due to this architecture, the encoder network iteratively learns an efficient data compression function, which maps the data to a lower dimensional representation. Following training, the decoder is able to successfully reconstruct the original input data, as the reconstruction error (*difference between input and reconstructed output produced by the decoder*) is the objective function throughout the training process.

## Implementation

Now that we understand the underlying architecture of an Autoencoder model, we can begin to implement the model.


The first step is to install the libraries, packages and modules which we shall use:

```python
# for data handling:
import numpy as np
import pandas as pd
from datetime import date, datetime

# for RNN Autoencoder:
from tensorflow import keras
from tensorflow.keras import layers

# for Plotting:
!pip install chart-studio
import plotly.graph_objects as go
```

Secondly, we need to obtain some data to analyse. This article uses the [Historic-Crypto][link3] package to obtain historical Bitcoin ```(‘BTC’)``` data from ‘2013–06–06’ to present day. The code below also generates the daily Bitcoin returns and intraday price volatility, prior to removing any rows of missing data and returning the first 5 rows of the DataFrame.

```python
# import the Historic Crypto package:
!pip install Historic-Crypto
from Historic_Crypto import HistoricalData

# obtain bitcoin data, calculate returns and intraday volatility:
dataset = HistoricalData(start_date = '2013-06-06',ticker = 'BTC').retrieve_data()
dataset['Returns'] = dataset['Close'].pct_change()
dataset['Volatility'] = np.abs(dataset['Close']- dataset['Open'])
dataset.dropna(axis = 0, how = 'any', inplace = True)
dataset.head()
```

Now that we have obtained some data, we should visually scan each series for potential outliers. The ```plot_dates_values``` function below enables the iterative plotting of each series contained within the DataFrame.

```python
def plot_dates_values(data_timestamps, data_plot):
  '''
  This function provides plotly plots of the input series.
  Arguments: 
          data_timestamps: the timestamp associated with each instance of data.
          data_plot: the series of data to be plotted.
  Returns:
          fig: displays a figure of the series with a slider and buttons.
  '''

  fig = go.Figure()
  fig.add_trace(go.Scatter(x = data_timestamps, y = data_plot,
                           mode = 'lines',
                           name = data_plot.name,
                           connectgaps=True))
  fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1 Years", step="year", stepmode="backward"),
            dict(count=2, label="2 Years", step="year", stepmode="backward"),
            dict(count=3, label="3 Years", step="year", stepmode="backward"),
            dict(label="All", step="all")
        ]))) 
  
  fig.update_layout(
    title=data_plot.name,
    xaxis_title="Date",
    yaxis_title="",
    font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  return fig.show()
```

We can now iteratively call the above function, generating Plotly charts for the Volume, Close, Open, Volatility and Return profiles of Bitcoin.

```python
plot_dates_values(dataset.index, dataset['Volume'])
```

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/324.embed"></iframe>

Notably, a number of spikes in trading volume occur in 2020, it may be useful to investigate if these spikes are anomalous or indicative of the broader series.

As such, we can begin data preprocessing for the Autoencoder model. The first step in data preprocessing is to determine an appropriate split between the training data and testing data. The ```generate_train_test_split``` function outlined below enables the splitting of training and testing data by date. Upon calling the below function, two DataFrames, namely ```training_data``` and ```testing_data``` are generated as global variables.

```python
def generate_train_test_split(data, train_end, test_start):
  '''
  This function splits the dataset into training data and testing data through
  use of strings. The strings provided as arguments for 'train_end' and
  'test_start' must be on sequential days.
  Arguments: 
          data: the DataFrame to be split into training and testing data.
          train_end: the date on which the training data ends (str).
          test_start: the date on which the testing data begins (str).
  Returns:
          training_data: data to be used in model training (Pandas DataFrame).
          testing_data: the data to be used in model testing (Pandas DataFrame).
  '''
  if isinstance(train_end, str) is False:
    raise TypeError("train_end argument should be a string.")
  
  if isinstance(test_start, str) is False:
    raise TypeError("test_start argument should be a string.")

  train_end_datetime = datetime.strptime(train_end, '%Y-%m-%d')
  test_start_datetime = datetime.strptime(test_start, '%Y-%m-%d')
  while train_end_datetime >= test_start_datetime:
    raise ValueError("train_end argument cannot occur prior to the test_start argument.")
  while abs((train_end_datetime - test_start_datetime).days) > 1:
    raise ValueError("the train_end argument and test_start argument should be seperated by 1 day.")

  training_data = data[:train_end]
  testing_data = data[test_start:]

  print('Train Dataset Shape:',training_data.shape)
  print('Test Dataset Shape:',testing_data.shape)

  return training_data, testing_data


# We now call the above function, generating training and testing data:
training_data, testing_data = generate_train_test_split(dataset, '2018-12-31','2019-01-01')
```

To improve model accuracy, we can ‘normalise’ or scale the data. This function scales the ```training_data``` DataFrame generated above, saving the ```training_mean``` and ```training_std``` for standardising the testing data later.


*Note: It is important to scale the training and testing data using the same scale, otherwise the difference in scale will generate interpretability issues and model inconsistencies.*

```python
def normalise_training_values(data):
  '''
  This function normalises the input values by both mean and standard deviation.
  The mean and standard deviation must be saved for test set standardisation downstream.
  Arguments: 
          data: the DataFrame column to be normalised.
  Returns:
          values: normalised data to be used in model training (numpy array).
          mean: the training set mean, to be used for normalising test set (float).
          std: the training set standard deviation, to be used for normalising the test set (float).
  '''
  if isinstance(data, pd.Series) is False:
    raise TypeError("data argument should be a Pandas Series.")

  values = data.to_list()
  mean = np.mean(values)
  values -= mean
  std = np.std(values)
  values /= std
  print("*"*80)
  print("The length of the training data is: {}".format(len(values)))
  print("The mean of the training data is: {}".format(mean.round(2)))
  print("The standard deviation of the training data is {}".format(std.round(2)))
  print("*"*80)
  return values, mean, std


# now call above function:
training_values, training_mean, training_std = normalise_training_values(training_data['Volume'])
```

As we have called the ```normalise_training_values``` function above, we now have a numpy array containing normalised training data called ```training_values```, and we have stored ```training_mean``` and ```training_std``` as global variables to be used in standardising the test set.

We can now begin to generate a series of sequences which can be used to train the Autoencoder model. We define that the model shall be provided with 30 previous observations, providing a 3D training data of the shape (2004,30,1):

```python
# define the number of time-steps in each sequence:
TIME_STEPS = 30

def generate_sequences(values, time_steps = TIME_STEPS):
  '''
  This function generates sequences of length 'TIME_STEPS' to be passed to the model.
  Arguments: 
          values: the normalised values which generate sequences (numpy array).
          time_steps: the length of the sequences (int).
  Returns:
          train_data: 3D data to be used in model training (numpy array).
  '''
  if isinstance(values, np.ndarray) is False:
    raise TypeError("values argument must be a numpy array.")
  if isinstance(time_steps, int) is False:
    raise TypeError("time_steps must be an integer object.")

  output = []

  for i in range(len(values) - time_steps):
    output.append(values[i : (i + time_steps)])
  train_data = np.expand_dims(output, axis =2)
  print("Training input data shape: {}".format(train_data.shape))

  return train_data
  
# now call the above function to generate x_train:  
x_train = generate_sequences(training_values)
```

Now that we have completed the training data processing, we can define the Autoencoder model, then fit the model on the training data. The ```define_model``` function utilises the training data shape to define an appropriate model, returning both the Autoencoder model, and a summary of the Autoencoder model.

```python
def define_model(x_train):
  '''
  This function uses the dimensions of x_train to generate an RNN model.
  Arguments: 
          x_train: 3D data to be used in model training (numpy array).
  Returns:
          model: the model architecture (Tensorflow Object).
          model_summary: a summary of the model's architecture.
  '''

  if isinstance(x_train, np.ndarray) is False:
    raise TypeError("The x_train argument should be a 3 dimensional numpy array.")

  num_steps = x_train.shape[1]
  num_features = x_train.shape[2]

  keras.backend.clear_session()
  
  model = keras.Sequential(
      [
       layers.Input(shape=(num_steps, num_features)),
       layers.Conv1D(filters=32, kernel_size = 15, padding = 'same', data_format= 'channels_last',
                     dilation_rate = 1, activation = 'linear'),
       layers.LSTM(units = 25, activation = 'tanh', name = 'LSTM_layer_1',return_sequences= False),
       layers.RepeatVector(num_steps),
       layers.LSTM(units = 25, activation = 'tanh', name = 'LSTM_layer_2', return_sequences= True),
       layers.Conv1D(filters = 32, kernel_size = 15, padding = 'same', data_format = 'channels_last',
                     dilation_rate = 1, activation = 'linear'),
       layers.TimeDistributed(layers.Dense(1, activation = 'linear'))
      ]
  )

  model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001), loss = "mse")
  return model, model.summary()
```

Subsequently, the ```model_fit``` function calls the ```define_model``` function internally, then provides ```epochs``` , ```batch_size``` and ```validation_loss``` parameters to the model. This function is then called, beginning the model training process.

```python
def model_fit():
  '''
  This function calls the above 'define_model()' function, subsequently training
  the model on the x_train data.
  Arguments: 
          N/A.
  Returns:
          model: the trained model.
          history: a summary of how the model trained (training error, validation error).
  '''
  # call the define_model function above on x_train:
  model, summary = define_model(x_train)

  history = model.fit(
    x_train,
    x_train,
    epochs=400,
    batch_size=128,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", 
                                              patience=25, 
                                              mode="min", 
                                              restore_best_weights=True)])
  
  return model, history


# call the above function, generating the model and the model's history:
model, history = model_fit()
```

Once the model has been trained, it is important to plot the training and validation loss curves to understand if the model suffers from bias (underfitting) or variance (overfitting). This can be observed through calling the below ```plot_training_validation_loss``` function.

```python
def plot_training_validation_loss():
  '''
  This function plots the training and validation loss curves of the trained model,
  enabling visual diagnosis of underfitting (bias) or overfitting (variance).
  Arguments: 
          N/A.
  Returns:
          fig: a visual representation of the model's training loss and validation
          loss curves.
  '''
  training_validation_loss = pd.DataFrame.from_dict(history.history, orient='columns')

  fig = go.Figure()
  fig.add_trace(go.Scatter(x = training_validation_loss.index, y = training_validation_loss["loss"].round(6),
                           mode = 'lines',
                           name = 'Training Loss',
                           connectgaps=True))
  fig.add_trace(go.Scatter(x = training_validation_loss.index, y = training_validation_loss["val_loss"].round(6),
                           mode = 'lines',
                           name = 'Validation Loss',
                           connectgaps=True))
  
  fig.update_layout(
  title='Training and Validation Loss',
  xaxis_title="Epoch",
  yaxis_title="Loss",
  font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  return fig.show()


# call the above function:
plot_training_validation_loss()
```

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/334.embed"></iframe>

Notably, both training and validation loss curves are converging throughout the chart, with the validation loss remaining slightly larger than the training loss. Given both the shape and relative errors, we can determine that the Autoencoder model does not suffer from underfitting or overfitting.

Now, we can define the reconstruction error, one of the core principles of the Autoencoder model. The reconstruction error is denoted as ```train_mae_loss```, with the reconstruction error threshold determined as the maximal value of ```train_mae_loss```. Consequently, when the test error is calculated, any value greater than the maximal value of ```train_mae_loss``` can be considered as an outlier.

```python
def reconstruction_error(x_train):
  '''
  This function calculates the reconstruction error and displays a histogram of
  the training mean absolute error.
  Arguments: 
          x_train: 3D data to be used in model training (numpy array).
  Returns:
          fig: a visual representation of the training MAE distribution.
  '''

  if isinstance(x_train, np.ndarray) is False:
    raise TypeError("x_train argument should be a numpy array.")

  x_train_pred = model.predict(x_train)
  global train_mae_loss
  train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis = 1)
  histogram = train_mae_loss.flatten() 
  fig =go.Figure(data = [go.Histogram(x = histogram, 
                                      histnorm = 'probability',
                                      name = 'MAE Loss')])  
  fig.update_layout(
  title='Mean Absolute Error Loss',
  xaxis_title="Training MAE Loss (%)",
  yaxis_title="Number of Samples",
  font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  
  print("*"*80)
  print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss).round(4)))
  print("*"*80)
  return fig.show()


# now call the above function:
reconstruction_error(x_train)
```

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/336.embed"></iframe>

Above, we saved the ```training_mean``` and ```training_std``` as global variables in order to use them for scaling test data. We now define the ```normalise_testing_values``` function for scaling the testing data.

```python
def normalise_testing_values(data, training_mean, training_std):
  '''
  This function uses the training mean and standard deviation to normalise
  the testing data, generating a numpy array of test values.
  Arguments: 
          data: the data to be used in model testing (Pandas DataFrame column).
          mean: the training set mean (float).
          std: the training set standard deviation (float).
  Returns:
          values: an array of testing values (numpy array).
  '''
  if isinstance(data, pd.Series) is False:
    raise TypeError("data argument should be a Pandas Series.")

  values = data.to_list()
  values -= training_mean
  values /= training_std
  print("*"*80)
  print("The length of the testing data is: {}".format(data.shape[0]))
  print("The mean of the testing data is: {}".format(data.mean()))
  print("The standard deviation of the testing data is {}".format(data.std()))
  print("*"*80)

  return values
```


Subsequently, this function is called on the ```Volume``` column of ```testing_data```. As such, the test_value is materialised as a numpy array.

```python
# now call the above function:
test_value = normalise_testing_values(testing_data['Volume'], training_mean, training_std) 
```

Following this, the ```generate_testing_loss``` function is defined, calculating the difference between the reconstructed data and the testing data. If any values are greater than the maximal value of ```train_mae_loss```, they are stored within the global anomalies list.

```python
def generate_testing_loss(test_value):
  '''
  This function uses the model to predict anomalies within the test set.
  Additionally, this function generates the 'anomalies' global variable,
  containing the outliers identified by the RNN.
  Arguments: 
          test_value: an array of testing values (numpy array).
  Returns:
          fig: a visual representation of the testing MAE distribution.
  '''
  x_test = generate_sequences(test_value)
  print("*"*80)
  print("Test input shape: {}".format(x_test.shape))

  x_test_pred = model.predict(x_test)
  test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis = 1)
  test_mae_loss = test_mae_loss.reshape((-1))

  global anomalies
  anomalies = (test_mae_loss >= np.max(train_mae_loss)).tolist()
  print("Number of anomaly samples: ", np.sum(anomalies))
  print("Indices of anomaly samples: ", np.where(anomalies))
  print("*"*80)

  histogram = test_mae_loss.flatten() 
  fig =go.Figure(data = [go.Histogram(x = histogram, 
                                      histnorm = 'probability',
                                      name = 'MAE Loss')])  
  fig.update_layout(
  title='Mean Absolute Error Loss',
  xaxis_title="Testing MAE Loss (%)",
  yaxis_title="Number of Samples",
  font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  
  return fig.show()


# call the above function:
generate_testing_loss(test_value)
```

Additionally, a distribution of the testing MAE loss is presented, for direct comparison with the training MAE loss.

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/338.embed"></iframe>

Finally, the outliers are visually represented below.

```python
def plot_outliers(data):
  '''
  This function determines the position of the outliers within the time-series,
  which are subsequently plotted.
  Arguments: 
          data: the initial dataset (Pandas DataFrame).
  Returns:
          fig: a visual representation of the outliers present in the series, as
          determined by the RNN.
  '''

  outliers = []

  for data_idx in range(TIME_STEPS -1, len(test_value) - TIME_STEPS + 1):
    time_series = range(data_idx - TIME_STEPS + 1, data_idx)
    if all([anomalies[j] for j in time_series]):
      outliers.append(data_idx + len(training_data))

  outlying_data = data.iloc[outliers, :]

  cond = data.index.isin(outlying_data.index)
  no_outliers = data.drop(data[cond].index)

  fig = go.Figure()
  fig.add_trace(go.Scatter(x = no_outliers.index, y = no_outliers["Volume"],
                           mode = 'markers',
                           name = no_outliers["Volume"].name,
                           connectgaps=False))
  fig.add_trace(go.Scatter(x = outlying_data.index, y = outlying_data["Volume"],
                           mode = 'markers',
                           name = outlying_data["Volume"].name + ' Outliers',
                           connectgaps=False))
  
  fig.update_xaxes(rangeslider_visible=True)

  fig.update_layout(
  title='Detected Outliers',
  xaxis_title=data.index.name,
  yaxis_title=no_outliers["Volume"].name,
  font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  
  
  return fig.show()


# call the final function:
plot_outliers(dataset)
```

The outlying data, as characterised by the Autoencoder model, are presented in orange, whilst conformant data is presented in blue.

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/340.embed"></iframe>

[link1]: <https://github.com/David-Woroniuk/Historic_Crypto>
[link2]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/RNN_Time_Series_Outlier_Detection.ipynb>
[link3]: <https://github.com/David-Woroniuk/Historic_Crypto>




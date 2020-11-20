---
title: 'Turtle Trading with Python— Is the trend really your friend?'
date: 2020-02-27 15:00:00
description: Article published by Data Driven Investor
featured_image: '/images/demo/turtletrading.jpg'
---

![](/images/demo/turtletrading.jpg)

# Turtle Trading with Python— Is the trend really your friend?

TL DR: [GitHub Code][link1]
##  So what is Turtle Trading?

In 1983, the legendary trader Richard Dennis, who had turned an initial stake of $5000 into $100Mn+, made a wager with business partner, William Eckhardt, that any individual could be taught to trade, or ‘grown’, in a similar way to the baby turtles he had observed in Singapore.
As such, ‘Turtle Trading’ was born, with the ‘turtles’, or students, provided with a series of rules which amounted to a complete trading system, encompassing the markets to trade, position-size, entries, exits and stop-losses.

## Main Components of Turtle Trading
The experiment aimed to provide an entirely mechanical approach, which may not be successful 100% of the time, but would provide rules which eliminate emotion and judgement, leaving the traders with the rules and nothing else:

*Markets* — What to buy or sell: The first decision is what to buy and sell, or essentially, what markets to trade.

*Position Sizing* — How much to buy or sell: The decision about how much to buy or sell is fundamental, and yet is often neglected by most traders.

*Entries* — When to buy or sell: The decision of when to buy or sell is called an entry decision.

*Exits* — When to get out of a winning position: Getting out of winning positions too early is one of the most common mistakes when trading trend following systems.

*Stop Losses* — When to get out of a losing position: The most important thing about reducing losses is to predefine the point where you will ‘stop out’ before you enter a position.

According to Richard Dennis, the rules are very simple, yet hard to apply in practice:
“I always say that you could publish my trading rules in the newspaper and no one would follow them. The key is consistency and discipline. Almost anybody can make up a list of rules that are 80% as good as what we taught our people. What they couldn’t do is give them the confidence to stick to those rules even when things are going bad.” – from Market Wizards, by Jack D. Schwager.

## Turtle Trading in Python
Let’s begin by importing the required package — Yahoo! Finance, whilst also importing the libraries Numpy, Pandas, Matplotlib and Datetime:

```python
# install required packages
!pip install yfinance --upgrade --no-cache-dir

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
```
Let’s use Datetime to generate a functional Dataset. Be careful, we have to use the Datetime argument .isoformat() to transform this into a value which can be interpreted by Yahoo! Finance:
```python
# find today's date
today = dt.date.today()
# delta = 1 day time delta
delta = dt.timedelta(days = 1)
# end_delta = 10 years ago
end_delta = dt.timedelta(days = 3652)

# Set the end of the sample to yesterday
end_of_sample = (today - delta)
# Set the start of the sample to 10 years ago
start_of_sample = (end_of_sample - end_delta)

# Change the data format from a 'datetime' element to a string which can be interpreted by Yahoo! Finance
end_of_sample = end_of_sample.isoformat()
start_of_sample = start_of_sample.isoformat()
```
We can now use Pandas Datareader to import S&P 500 (^GSPC) Data for the specified sample:
```python
# use Datreader to import S&P500 Data within the defined sample period
from pandas_datareader import data as pdr
dataset = pdr.get_data_yahoo("^GSPC", start=start_of_sample, end=end_of_sample).reset_index()
```
Now, we need to generate some triggers under which the action of a ‘buy order’ or ‘sell order’ are initiated:
```python
# define the conditions under which the algorithm should trigger a buy order or sell order
count = int(np.ceil(len(dataset) * 0.1))

action = pd.DataFrame(index=dataset.index)
action['trigger'] = 0.0
action['trend'] = dataset['Adj Close']

action['RollingMax'] = (action.trend.shift(1).rolling(count).max())
action['RollingMin'] = (action.trend.shift(1).rolling(count).min())

action.loc[action['RollingMax'] < action.trend, 'trigger'] = -1
action.loc[action['RollingMin'] > action.trend, 'trigger'] = 1
```
We can call the ‘trigger’ column of the action data frame to check when actions are recommended to take place. However, this algorithm doesn’t accommodate net-short positions — if the inventory is 0, we cannot sell the Index. Now, we can define a decision function:
```python
# this executable is a decision function for the algorithm
def trade(price_change, trigger, capital = 10_000, maximum_long = 1, maximum_short = 1,):
    """
    price_change = S&P500 price change (Absolute Value)
    trigger = 1 initiates a buy order, -1 initiates sell order
    capital = The initial capital committed to algorithm (the _ acts as a comma for large values)
    maximum_long = the maximum quantity that can be purchased in any one action
    maximum_short = maximum quantity that can be sold in any one action (note the shortselling restriction)
    """
    starting_capital = capital
    sell_states = []
    buy_states = []
    inventory = 0

    def buy(i, capital, inventory):
        shares = capital // price_change[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough capital to buy a unit price %f'
                % (i, capital, price_change[i])
            )
        else:
            if shares > maximum_long:
                buy_units = maximum_long
            else:
                buy_units = shares
            capital -= buy_units * price_change[i]
            inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * price_change[i], capital)
            )
            buy_states.append(0)
        return capital, inventory

    for i in range(price_change.shape[0] - int(0.025 * len(dataset))):
        state = trigger[i]
        if state == 1:
            capital, inventory = buy( i, capital, inventory)
            buy_states.append(i)
        elif state == -1:
            if inventory == 0:
                    print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if inventory > maximum_short:
                    sell_units = maximum_short
                else:
                    sell_units = inventory
                inventory -= sell_units
                total_sell = sell_units * price_change[i]
                capital += total_sell
                try:
                    invest = (
                        (price_change[i] - price_change[buy_states[-1]])
                        / price_change[buy_states[-1]]) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (i, sell_units, total_sell, invest, capital))
            sell_states.append(i)
            
    invest = ((capital - starting_capital) / starting_capital) * 100
    total_gains = capital - starting_capital
    return buy_states, sell_states, total_gains, invest
```
When we execute this function, no output is obtained. This is because we must pass arguments to the decision function:
```python
# we must pass arguments to the decision function
buy_states, sell_states, total_gains, invest = trade(dataset['Adj Close'], action['trigger'])
```
Now that we have passed the arguments to our ‘trade’ function, each action that the code attempts to perform is printed, along with a corresponding outcome. Before we visualise the algorithm’s output, we can define a function called Index_Returns, which provides the simple return of the Index throughout the sample period.
The first Adjusted Close value and final Adjusted Close value are passed to the function, generating a Benchmark to measure the algorithm’s performance:
```python
# Let's define a quick function which generates the Index returns, to allow benchmarking
def Index_Returns(start_value, end_value):
    return (((end_value - start_value) / start_value) -1) * 100

# We can pass this function the required arguments to calculate a simple return
SP_500_Returns = Index_Returns(dataset['Adj Close'].iloc[0], dataset['Adj Close'].iloc[-1])
```
We can add the Benchmark performance to the Chart Title, and visualise the algorithm’s performance over the sample:
```python
# Plot the S&P500 Closing Price
value = dataset['Adj Close']
fig = plt.figure(figsize = (15,5))
plt.plot(value, color='b', lw=2.)

# Plot the Entry and Exit Signals generated by the algorithm
plt.plot(value, '^', markersize=8, color='g', label = 'Trigger Entry', markevery = buy_states)
plt.plot(value, 'v', markersize=8, color='r', label = 'Trigger Exit', markevery = sell_states)

# Chart Title displaying the Absolute Returns, Return on Capital & Benchmark Returns
plt.title('Absolute Returns: $%f, Return on Capital: %f%%, Benchmark Return: %f%%'%(round(total_gains,2), round(invest,2), round(SP_500_Returns,2)))
plt.legend()
plt.show();
```
In summary, this algorithm does still generate a positive return over the sample, however, it considerably under-performs the benchmark. This could be due to the usage of low frequency data, but it is more likely that the parameters require tuning.

Stay tuned as we continue to code other trading algorithms!

[link1]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/Turtle_Trading_with_Python—_Is_the_trend_really_your_friend%3F.ipynb>
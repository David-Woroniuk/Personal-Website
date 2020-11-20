---
title: “Sentimentum Investing” — Combining Sentiment Analysis and Systematic Trading
date: 2020-10-01 18:00:00
description: Article published by The Startup.
featured_image: '/images/demo/sentimentum.jpg'
---

![](/images/demo/sentimentum.jpg)

# “Sentimentum Investing” — Combining Sentiment Analysis and Systematic Trading

*Utilising 250,000+ Tweets to backtest “sentimentum” trading strategies*

TL DR: [Data][link1], [Code][link2], [GitHub][link3].

The Financial sector generates a huge volume of data each day, with Google processing over 3.5 Billion searches per day. This data comes in many forms; Factual news, Scheduled Economic releases, Company filings and Investor opinions.

Due to the continual generation of new information and opinions, traders often struggle to stay up to date manually, preferring to automate analysis and use the outputs to generate systematic trading strategies. This article provides a walkthrough of how to augment Simple Moving Average (SMA) trading strategies with *V*alence *A*ware *D*ictionary and s*E*ntiment *R*easoner (VADER) sentiment analysis, producing “Sentimentum” trading strategies.

## What is Sentiment Analysis?

Sentiment Analysis is a sub-category of Natural Language Processing (NLP), which aims to detect polarity (*ie. positive and negative opinions*) within a provided text. In essence, Sentiment Analysis measures the attitude, sentiment and emotions presented within a text sample, returning continuous values corresponding to positive, negative, or neutral scores.

## What is VADER?

VADER (*V*alence *A*ware *D*ictionary and s*E*ntiment *R*easoner) is a sentiment analysis model which is sensitive to both polarity (*ie. positive and negative opinions*) and intensity (*ie. strength of opinions*). As such, it can be thought of as similar to ‘one-hot encoding’ text (turning the categorical text variable into continuous values).

Notably, the VADER sentiment analysis toolkit is specifically tuned for social media sentiment, as opposed to more formal documents, accounting for textual phenomena such as ```‘!!!’``` or ```‘CAPS’```, which increase sentiment intensity.

Due to the above features, this article utilises VADER to quantify sentiment expressed within 250,000+ tweets containing the terms ‘Tesla’, ‘TESLA’ or ‘TSLA’ over a 1 week period, displaying the role which sentiment analysis can play in generating profitable trading strategies.

## Implementation

Now we have an understanding of the underlying model, we can begin to backtest some trading strategies.

The first step is to install the libraries, packages and modules which we shall use:
```python
# for data wrangling:
import pandas as pd
import numpy as np
import time as tm
import datetime as dt

# for retrieval of market data:
!pip install alpha_vantage
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries


# for natural language processing:
import nltk
import nltk.data
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
import re
import en_core_web_sm
nlp = en_core_web_sm.load()


# for plotting:
import matplotlib.pyplot as plt
```

Secondly, we need to obtain the tweets to analyse. For brevity, this article provides pre-cleaned twitter data, which can be found [here][link4]. The below code imports this data and visualises the first 10 rows of the DataFrame.

```python
# import pre-cleaned dataset from GitHub:
dataset = pd.read_csv('https://raw.githubusercontent.com/David-Woroniuk/Medium-Articles/master/twitter_data.csv', index_col = 'date', infer_datetime_format= 'date')
dataset.head(10)
```

Following this, we need to obtain the corresponding ‘TSLA’ market data. This walkthrough uses Alpha Vantage’s great [free API][link5] to obtain market data on a minute frequency. The below code accesses the API, manipulates the data and displays the first 5 rows of ```market_data```.

```python
# define the sample period of market data:
today = dt.datetime.now()
delta = dt.timedelta(days = 1)
end_delta = dt.timedelta(days = 9)
end_of_sample = (today - delta)
start_of_sample = (end_of_sample - end_delta)

start_of_sample = start_of_sample.replace(second=0,microsecond=0)
end_of_sample = end_of_sample.replace(second=0,microsecond=0)

# define a Dataframe using the index as our defined sample:
market_data = pd.DataFrame(index=pd.date_range(start=start_of_sample,end=end_of_sample,freq = '1min'))

# call alphavantage's API:
ts = TimeSeries(key='YOUR API KEY HERE', output_format='pandas')
data, meta_data = ts.get_intraday(symbol= 'TSLA', interval='1min', outputsize='full')

# place data into market_data df:
market_data = pd.concat([market_data,data],axis=1)
market_data.index = market_data.index.strftime("%d/%m/%Y %H:%M")
market_data.dropna(axis = 0, how = 'any', inplace = True)
market_data.head()
```

After this, we can combine the financial data held within ```market_data``` with the twitter data held within ```dataset``` , and re-label the columns.

```python
# now merge twitter data and market data:
market_data = market_data.merge(dataset, left_index=True, right_index=True, how='inner')

# rename the columns for ease:
market_data.rename(columns={"1. open": "open",
                            "2. high": "high",
                            "3. low" : "low",
                            "4. close" : "close",
                            "5. volume" : "volume",
                            "content" : "tweet_content"}, inplace = True)
```

Next, we remove any additional regular expression (regex) characters which represent line feed characters ```(\n)```, carriage return characters ```(\r )```, tabs ```(\t)``` or no break space ```(\xa0)``` in order to clean the data.

```python
# remove additional regex characters from twitter data:
for i in range(len(market_data)):
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\n', ' ')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\r', ' ')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\t', '')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\xa0', '')
```

Following this, we define and apply some functions to the tweets, determining the twitter handles of retweets and mentions, and any hashtags mentioned. This data could be further analysed through [Named Entity Recognition (NER)][link6] techniques or application of a ‘Bag of Words’ focussing on specific search terms.

```python
# define functions to determine retweets, mentions and hashtags: 
def find_retweeted(tweet):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet) 


# apply functions to market_data, generate output columns:
market_data['retweeted'] = market_data['tweet_content'].apply(find_retweeted)
market_data['mentioned'] = market_data['tweet_content'].apply(find_mentioned)
market_data['hashtags'] = market_data['tweet_content'].apply(find_hashtags)
```

Now that the dataset has been combined and cleaned, we can apply the VADER sentiment analysis. We initialise empty lists, then iterate through each row of the dataset, appending compound sentiment scores, positive, neutral and negative sentiment scores to the corresponding lists. This information is subsequently added to the DataFrame.

```python
# load nltk vader sentiment analysis as analyzer:
analyzer = SentimentIntensityAnalyzer()

# define empty lists:
compound_sentiment = []
vs_pos = []
vs_neu = []
vs_neg = []

# for each row in 'tweet_content', analyze sentiment:
for i in range(0, len(market_data)):
  compound_sentiment.append(analyzer.polarity_scores(market_data['tweet_content'][i])['compound'])
  vs_pos.append(analyzer.polarity_scores(market_data['tweet_content'][i])['pos'])
  vs_neu.append(analyzer.polarity_scores(market_data['tweet_content'][i])['neu'])
  vs_neg.append(analyzer.polarity_scores(market_data['tweet_content'][i])['neg'])


# generate output columns:
market_data['total_sentiment'] = compound_sentiment
market_data['positive'] = vs_pos
market_data['neutral'] = vs_neu
market_data['negative'] = vs_neg
```

## Backtesting

The code required for the long only backtest is outlined below. All trading strategies are provided with $10,000 of capital, allowed to purchase or sell one stock per minute, and constrained from short-selling. Additionally, each strategy is provided with a figure, visualising the time-stamp at which purchase and sale decisions were made.

```python
def trade(data, price_change, trigger, capital = 10_000, maximum_long = 1, maximum_short = 1):
    '''
    price_change = market price change.
    trigger = 1 is a buy order, -1 is sell order.
    capital = initial capital committed to algorithm.
    maximum_long = maximum quantity that can be purchased in one period.
    maximum_short = maximum quantity that can be sold in one period.
    '''
    starting_capital = capital
    sell_states = []
    buy_states = []
    inventory = 0

    def buy(i, capital, inventory):
        shares = capital // price_change[i]
        if shares < 1:
            print('{}: total balance {}, not enough capital to buy a unit price {}'.format(data.index[i], capital, price_change[i]))
        else:
            if shares > maximum_long:
                buy_units = maximum_long
            else:
                buy_units = shares
            capital -= buy_units * price_change[i]
            inventory += buy_units
            print('{}: buy {} units at price {}, total balance {}'.format(data.index[i], buy_units, buy_units * price_change[i], capital))
            buy_states.append(0)
        return capital, inventory


    
    for i in range(price_change.shape[0] - int(0.025 * len(price_change))):
        state = trigger[i]
        if state == 1:
            capital, inventory = buy( i, capital, inventory)
            buy_states.append(i)
        elif state == -1:
            if inventory == 0:
                    print('{}: cannot sell anything, inventory 0'.format(data.index[i]))
            else:
                if inventory > maximum_short:
                    sell_units = maximum_short
                else:
                    sell_units = inventory
                inventory -= sell_units
                total_sell = sell_units * price_change[i]
                capital += total_sell
                try:
                    RoC = ((price_change[i] - price_change[buy_states[-1]]) / price_change[buy_states[-1]]) * 100
                except:
                    RoC = 0
                print('{}, sell {} units at price {}, RoC: {}%, total balance: {}'.format(data.index[i], sell_units, total_sell, RoC, capital))
            sell_states.append(i)

    RoC = ((capital - starting_capital) / starting_capital) * 100
    total_gains = capital - starting_capital
    consolidated_position = (capital + inventory * price_change[i])
    print('*'*150)
    print("Consolidated Position:{}, Realised Gains:{}, Realised Return on Capital:{}, Inventory:{}".format(consolidated_position, total_gains, RoC, inventory))
    print('*'*150)


    # Plotting:
    value = data['close']
    fig = plt.figure(figsize = (20,10))
    plt.plot(value, color = 'b', lw=2.)

    # Plot the Entry and Exit Signals generated by the algorithm:
    plt.plot(value, '^', markersize=8, color='g', label = 'Trigger Entry', markevery = buy_states)
    plt.plot(value, 'v', markersize=8, color='r', label = 'Trigger Exit', markevery = sell_states)

    # Chart Title displaying the Absolute Returns, Return on Capital & Benchmark Returns:
    plt.title('Consolidated Position: {}, Realised Absolute Returns: {}, Realised Return on Capital: {}%, Inventory: {}'.format(round(consolidated_position,2), round(total_gains,2), round(RoC,2), inventory))
    plt.legend()
    plt.show()

    return buy_states, sell_states, total_gains, RoC
```

## Systematic Trading Strategies

*Strategy 1*: A Simple Moving Average (SMA) strategy. This strategy utilises the 21min MA and 50min MA, buying when the 21min MA is larger than the 50min MA, and selling if the opposite is true. However, this strategy is constrained as ‘long only’ in the backtest below.

*Results*: Consolidated Position: $9885.127, Realised Gains: $-114.88, Inventory: 0

```python
# copy market_data to maintain data integrity:
strategy_one = market_data.copy(deep = True)

# devise simple moving average strategy:
strategy_one['21_SMA'] = strategy_one['close'].rolling(window = 21).mean()
strategy_one['50_SMA'] = strategy_one['close'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_one = strategy_one[strategy_one['21_SMA'].notna()]
strategy_one['trigger'] = 0

# define trading triggers:
strategy_one.loc[(strategy_one['21_SMA'] < strategy_one['50_SMA']), 'trigger'] = -1
strategy_one.loc[(strategy_one['21_SMA'] > strategy_one['50_SMA']), 'trigger'] = 1

buy_states, sell_states, total_gains, RoC = trade(strategy_one, strategy_one['close'], strategy_one['trigger'])
```

*Strategy 2*: An SMA strategy applied to total sentiment. This strategy utilises the 21min MA and 50min MA of ```total_sentiment```, buying when the 21min MA is larger than the 50min MA, and selling if the opposite is true. Whilst this is a long only strategy in a down market, it outperforms buy and hold strategies assuming sale of inventory at the end of the sample.

*Results*: Consolidated Position:$10379.12, Realised Gains:$-9660.08, Inventory: 24

```python
# copy market_data to maintain data integrity:
strategy_two = market_data.copy(deep=True)

# devise a sentiment moving average strategy:
strategy_two['21_SMA_Sentiment'] = strategy_two['total_sentiment'].rolling(window = 21).mean()
strategy_two['50_SMA_Sentiment'] = strategy_two['total_sentiment'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_two = strategy_two[strategy_two['21_SMA_Sentiment'].notna()]
strategy_two['trigger'] = 0

# define trading triggers:
strategy_two.loc[(strategy_two['21_SMA_Sentiment'] < strategy_two['50_SMA_Sentiment']), 'trigger'] = -1
strategy_two.loc[(strategy_two['21_SMA_Sentiment'] > strategy_two['50_SMA_Sentiment']), 'trigger'] = 1


buy_states, sell_states, total_gains, RoC = trade(strategy_one, strategy_one['close'], strategy_one['trigger'])
```

*Strategy 3*: An SMA strategy applied to positive and negative sentiment, as determined by VADER. This strategy utilises the 21min MA and 50min MA of ```total_sentiment```, ```positive sentiment``` and ```negative sentiment```, buying when the 21min MAs of total_sentiment and positive are is larger than the 50min MA and the 21min MA of negative is lower than the 50min MA, and selling if the opposite is true. Whilst this strategy underperforms Strategy 2, it incurs far fewer transaction costs, owing to the increased number of parameters.

*Results*: Consolidated Position: $9772.55, Realised Gains:$-4828.74, Inventory: 11

```python
# copy market_data to maintain data integrity:
strategy_three = market_data.copy(deep=True)

# devise sentiment variables:
strategy_three['21_SMA_Sentiment'] = strategy_three['total_sentiment'].rolling(window = 21).mean()
strategy_three['50_SMA_Sentiment'] = strategy_three['total_sentiment'].rolling(window = 50).mean()
strategy_three['21_SMA_Positive'] = strategy_three['positive'].rolling(window = 21).mean()
strategy_three['50_SMA_Positive'] = strategy_three['positive'].rolling(window = 50).mean()
strategy_three['21_SMA_Negative'] = strategy_three['negative'].rolling(window = 21).mean()
strategy_three['50_SMA_Negative'] = strategy_three['negative'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_three = strategy_three[strategy_three['21_SMA_Sentiment'].notna()]
strategy_three['trigger'] = 0

# define trading triggers:
strategy_three.loc[(strategy_three['21_SMA_Sentiment'] < strategy_three['50_SMA_Sentiment']) & 
                   (strategy_three['21_SMA_Positive'] < strategy_three['50_SMA_Positive']) &
                   (strategy_three['21_SMA_Negative'] > strategy_three['50_SMA_Negative']), 'trigger'] = -1


strategy_three.loc[(strategy_three['21_SMA_Sentiment'] > strategy_three['50_SMA_Sentiment']) & 
                   (strategy_three['21_SMA_Positive'] > strategy_three['50_SMA_Positive']) &
                   (strategy_three['21_SMA_Negative'] < strategy_three['50_SMA_Negative']), 'trigger'] = 1


buy_states, sell_states, total_gains, RoC = trade(strategy_three, strategy_three['close'], strategy_three['trigger'])
```

*Strategy 4*: A combination of Strategies 1 & 3. We utilise the SMAs of price to generate a momentum strategy, combined with the sentiment based momentum strategy outlined in Strategy 3, to produce a “Sentimentum” strategy.

*Results*: Consolidated Position: $9949.29, Realised Gains:$-2560.53, Inventory: 6

```python
# copy market_data to maintain data integrity:
strategy_four = market_data.copy(deep=True)

# devise sentiment variables:
strategy_four['21_SMA'] = strategy_four['close'].rolling(window = 21).mean()
strategy_four['50_SMA'] = strategy_four['close'].rolling(window = 50).mean()

strategy_four['21_SMA_Sentiment'] = strategy_four['total_sentiment'].rolling(window = 21).mean()
strategy_four['50_SMA_Sentiment'] = strategy_four['total_sentiment'].rolling(window = 50).mean()

strategy_four['21_SMA_Positive'] = strategy_four['positive'].rolling(window = 21).mean()
strategy_four['50_SMA_Positive'] = strategy_four['positive'].rolling(window = 50).mean()

strategy_four['21_SMA_Negative'] = strategy_four['negative'].rolling(window = 21).mean()
strategy_four['50_SMA_Negative'] = strategy_four['negative'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_four = strategy_four[strategy_four['50_SMA'].notna()]
strategy_four['trigger'] = 0

# # define trading triggers:
strategy_four.loc[(strategy_four['21_SMA'] < strategy_four['50_SMA']) &
                  (strategy_four['21_SMA_Sentiment'] < strategy_four['50_SMA_Sentiment']) & 
                  (strategy_four['21_SMA_Positive'] < strategy_four['50_SMA_Positive']) &
                  (strategy_four['21_SMA_Negative'] > strategy_four['50_SMA_Negative']), 'trigger'] = -1


strategy_four.loc[(strategy_four['21_SMA'] > strategy_four['50_SMA']) &
                  (strategy_four['21_SMA_Sentiment'] > strategy_four['50_SMA_Sentiment']) & 
                  (strategy_four['21_SMA_Positive'] > strategy_four['50_SMA_Positive']) &
                  (strategy_four['21_SMA_Negative'] < strategy_four['50_SMA_Negative']), 'trigger'] = 1


buy_states, sell_states, total_gains, RoC = trade(strategy_four, strategy_four['close'], strategy_four['trigger'])
```


This walkthrough outlines a few potential systematic trading strategies, however it doesn’t account for short-selling or volatility considerations, which represent interesting areas of further in-sample optimisation. Further to this, additional parameters may reduce the impact of transaction costs on strategy profitability.

Feel free to access the data and code within the GitHub repo and develop your own strategies. 

Happy Coding!


[link1]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/twitter_data.csv>
[link2]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/Sentimentum_Investing.py>
[link3]: <https://github.com/David-Woroniuk/Medium-Articles>
[link4]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/twitter_data.csv>
[link5]: <https://www.alphavantage.co/>
[link6]: <https://towardsdatascience.com/fomc-named-entity-recognition-has-information-content-evolved-2da6da8abd88>
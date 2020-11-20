---
title: 'Time-Series Animation in Matplotlib'
date: 2020-03-19 15:00:00
description: Article published by Data Driven Investor
featured_image: '/images/demo/turtletrading.jpg'
---

![](/images/demo/timeseries.jpg)

# Time-Series Animation in Matplotlib
*Use the Matplotlib library to animate time-series data.*

TL DR: [GitHub Code.][link1]

Animations are an interesting way of demonstrating time-series data such as financial products, climate change, seasonal sales patterns and social media trends, as we can observe how the data evolves over time.

This article will provide a walkthrough of how to animate time-series data, demonstrating the price convergence of Henry Hub (USA) and TTF (Netherlands).

So, let’s go ahead and install the required packages:
```python
# pip install Quandl - we shall use Quandl to download the Data:
!pip install quandl
# now import the Quandl package:
import quandl
# set an output directory for the animation and chart:
root_dir = '/content/drive/My Drive/'
# provide Quandl with an API Key:
quandl.ApiConfig.api_key = 'YOUR API KEY HERE'
```

We have now imported the Quandl package which enables access to the datasets provided by [Quandl.com][link2]. We can import more packages to work with the data and develop an animation:
```python
# import datetime to help define sample period:
import datetime as dt
# import pandas for data wrangling:
import pandas as pd
# import matplotlib, pyplot and animation for plotting and animating:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define the animation embed limit for matplotlib:
matplotlib.rcParams['animation.embed_limit'] = 200**128
```

Now, we will define the sample period that we wish to work with. For this project, I chose a sample of 712 days:

```python
# Find today's date & set delta to 712 days ago:
today = dt.date.today()
delta = dt.timedelta(days = 712)

# Set the end of the sample period to today, start of the sample period to 712 days ago:
end_of_sample = today
start_of_sample = (today - delta)

# Change the data format from a 'datetime' element to a string which can be read by Quandl's API:
end_of_sample = end_of_sample.isoformat()
start_of_sample = start_of_sample.isoformat()
```
We now need to collect the data from Quandl. Notably, the Henry Hub prices are expressed in $/MMBTu, whereas the TTF prices are expressed in €/MWh. Consequently, we have to transform the data to be expressed in the same units:
```python
# Set an empty pandas DataFrame using the defined sample period as index, select 'B' for business days:
Dataframe = pd.DataFrame(index=pd.date_range(start=start_of_sample,end=end_of_sample,freq = 'B'))
# Call the Quandl API for Henry Hub price data for the sample period:
henry_hub_price_data = quandl.get('CHRIS/CME_NG1', start_date= start_of_sample, end_date= end_of_sample, paginate=True)
# Call the Quandl API for TTF price data for the sample period:
ttf_price_data = quandl.get("CHRIS/ICE_TFM1", start_date= start_of_sample, end_date= end_of_sample, paginate=True)
# Call the Quandl API for €/$ price data for the sample period:
euro_dollar = quandl.get("ECB/EURUSD", start_date= start_of_sample, end_date= end_of_sample, paginate=True )
# transform the ttf data from €/MWh to $/MMBTu, using the conversion rate of 1MWh : 3.4121MMBTu:
ttf_dollar_mmbtu = round(((euro_dollar['Value'] * ttf_price_data['Settle'])/3.4121),3)
# Concatenate the Settlement price at Henry Hub and the dollar transformed TTF Settlement price to Dataframe:
Dataframe = pd.concat([Dataframe,henry_hub_price_data['Settle'],ttf_dollar_mmbtu],axis=1)
# Rename the columns of Dataframe:
Dataframe.columns = ['Henry Hub Continuous Futures','TTF Continuous Futures']
# Remove any na values by row:
Dataframe = Dataframe.dropna(axis=0, how='any')
```

Now we have the data, let’s generate a quick static plot to visualise the dataset:
```python
# Set the notebook to display matplotlib charts inline:
%matplotlib inline
# set a figure of size (15,5):
fig = plt.figure(figsize = (15,5))
# add limits on the x axis defined by the sample period (0 is the first observation, -1 the final observation):
plt.xlim(Dataframe.index[0],Dataframe.index[-1])
# add limits on the y-axis defined by minimum and maximum of the respective series, incorporate some additional room:
plt.ylim((Dataframe['Henry Hub Continuous Futures'].min()-0.1), (Dataframe['TTF Continuous Futures'].max()+0.1))
# plot the Henry Hub values with a dashed blue line, width 2:
plt.plot(Dataframe['Henry Hub Continuous Futures'], data= Dataframe, marker='', color='blue', linewidth = 2, linestyle = 'dashed')
# plot the TTF values with a red line, width 2:
plt.plot(Dataframe['TTF Continuous Futures'], data= Dataframe, marker='', color='red', linewidth=2)
# set the plot title:
plt.title('Henry Hub Continuous Futures & TTF Continuous Futures', fontsize=14)
# set the x-axis label:
plt.xlabel('Time',fontsize=10)
# set the y-axis label:
plt.ylabel('Price ($/MMBtu)',fontsize=10)
# add a legend to the plot:
plt.legend()
# save the output to the pre-defined output directory:
plt.savefig(root_dir + 'Henry Hub vs TTF.png')
# show the chart:
plt.show();
```

Great, we have plotted a static chart containing our data series. Now, let’s move to the animations. Firstly, we need to set up the axes for the dynamic chart:
```python
# set a figure of size (10,6):
fig = plt.figure(figsize=(10,6))
# set subplot grid parameters (1x1 grid, 1st subplot):
ax1 = fig.add_subplot(1,1,1)
# add limits on the x axis defined by the sample period (0 is the first observation, -1 the final observation):
ax1.axis(xmin = Dataframe.index[0], xmax = Dataframe.index[-1])
# add limits on the y-axis defined by minimum and maximum of the respective series, incorporate some additional room:
ax1.axis(ymin= (Dataframe['Henry Hub Continuous Futures'].min()-0.1),ymax=(Dataframe['TTF Continuous Futures'].max()+0.1))
```

Next, we define a function, animate, which requires the input argument of i. This function can be called multiple times to construct the animation.

```python
# define the function animate, which has the input argument of i:
def animate(i):
#   set the variable data to contain 0 to the (i+1)th row:
  data =  Dataframe.iloc[:int(i+1)]  #select data range
  #   initialise xp as an empty list:
  xp = []
  #   initialise yp as an empty list:
  yp = []
  #   initialise zp as an empty list:
  zp = []
  
#   set the variable lines as equal to the variable data:
  lines = data

#   for a line in lines:
  for line in lines:
    #     x is equal to the index (time domain):
    xp = data.index
    #     y is equal to the 'Henry Hub Continuous Futures' column
    yp = data['Henry Hub Continuous Futures']
    #     z is equal to the 'TTF Continuous Futures' column
    zp = data['TTF Continuous Futures']

  #   clear ax(1):
  ax1.clear()
  
  #   add a textbox in the top right corner (1, 0.9):
  ax1.text(1, 0.9, 'by David Woroniuk', transform=ax1.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.1, edgecolor='white'))
  #   plot Henry Hub Continual Futures:
  ax1.plot(xp, yp)
  #   plot TTF Continual Futures:   
  ax1.plot(xp, zp)

  #   provide a label for the x-axis:
  plt.xlabel('Time',fontsize=12)
  #   provide a label for the y-axis:  
  plt.ylabel('Price $/MMBtu',fontsize=12)
  #   provide a plot title:   
  plt.title('Henry Hub vs TTF Futures',fontsize=14)
 ```
 
 We can call the ```matplotlib.animation.FuncAnimation``` function, providing input arguments of a figure, the animate function which we just defined, and frames, which specifies how many times the animation function should be called. Interval provides a delay between frames and is measured in milliseconds.
As I work in Google Colab, I chose to define the animation writer as ```[‘ffmpeg’]```, however, many IPython shells work well with ImageMagick.

```python
# call Matplotlib animation.Funcanimation, providing the input arguments of fig, animate, the number of frames and an interval:
ani = animation.FuncAnimation(fig, animate, frames = len(Dataframe), interval=10) 

# Use the 'ffmpeg' writer:
Writer = animation.writers['ffmpeg']
# Set the frames per second and bitrate of the video:
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# save the animation to the predefined output directory:
ani.save(root_dir +'animation_video.mp4', writer=writer)
```

Voilà! We now have a time-series animation saved within the root_dir!

[link1]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/Time_Series_Animation_in_Matplotlib.ipynb>
[link2]: <https://www.quandl.com/>

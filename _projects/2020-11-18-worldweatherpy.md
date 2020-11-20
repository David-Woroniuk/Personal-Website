---
title: WorldWeatherPy — a Python package for weather data extraction
date: 2020-11-18 18:00:00
description: Usage notes for the WorldWeatherPy Python package.
featured_image: '/images/demo/worldweatherpy.jpg'
---

![](/images/demo/worldweatherpy.jpg)

# WorldWeatherPy — a Python package for weather data extraction

*Usage notes for the WorldWeatherPy package*

TL DR: [GitHub][link1] [PYPI Package][link2]


As a researcher focused on the Energy sector, weather variables are typically amongst the most important features of regression models. Despite the diverse applications of weather data in Data Science, I was unable to find a Python package which enables full customisation of weather data retrieval.

As Data Science projects typically require specific variables for a large number of cities, downloading all the available data, followed by removal of redundant variables for each city, reduces the data extraction and processing speed. As such, a fully customisable wrapper is required.

## World Weather Online

I chose the [WorldWeatherOnline API][link3] for this project, for a number of reasons:

1) *Free sign up (for the trial period).*
No credit card details are required for the trial period, which enables access to the full range of API capabilities. This enables access to 500 requests per key each day, which enables the retrieval of a substantial volume of weather data.

2) *Fully customisable.*
The API can be called using a range of data frequencies and has a high level of global coverage.

3) *Data Consistency.*
The data is consistent with national weather monitoring websites, which may otherwise require payment for API use.

The [WorldWeatherOnline][link3] website has an interactive page for retrieval of JSON or XML formats, which can be accessed [here][link4]. However, these outputs require processing, increasing development time for downstream projects. Due to this, I wrote the [WorldWeatherPy][link5] package, which handles all the required input formatting and output processing, returning Pandas DataFrames, or CSV files.

## WorldWeatherPy

The [WorldWeatherPy][link5] package can be used to extract historical weather data from the WorldWeatherOnline API, parsing the data into DataFrame or CSV formats.

This section shall provide a brief walkthrough of the input arguments and functionalities of the WorldWeatherPy package.The below code block enables package installation, and importation of the modules.
```python
pip install WorldWeatherPy

from WorldWeatherPy import DetermineListOfAttributes
from WorldWeatherPy import HistoricalLocationWeather
from WorldWeatherPy import RetrieveByAttribute
```

If you are unsure of the attributes available through the WorldWeatherOnline API, a complete list can be printed using the ```retrieve_list_of_options``` function of the ```DetermineListOfAttributes``` module:

```python
attributes = DetermineListOfAttributes(api_key, verbose = True).retrieve_list_of_options()
```

If you require standard weather information about a city or location, the ```retrieve_hist_data``` function of the ```HistoricalLocationWeather``` module can be called, with the required user inputs outlined below:
```python
dataset = HistoricalLocationWeather(api_key, city, start_date, end_date, frequency).retrieve_hist_data()
```

If you require full customisation of the weather attributes retrieved, the ```retrieve_hist_data``` function of the ```RetrieveByAttribute``` module can be called, although this requires a list of weather attributes (attribute_list) as an input argument.
```python
dataset = RetrieveByAttribute(api_key, attribute_list, city, start_date, end_date, frequency).retrieve_hist_data()
```

In addition to the required input arguments, two optional input arguments, ```verbose``` and ```csv_directory``` can be added. The ```verbose``` argument defaults to ```True```, printing progress reports throughout execution of the script, whilst the ```csv_directory``` argument enables specification of an output directory, to which the output data is saved as a .csv file.

All input arguments for the modules and a short description can be found on the [GitHub repository][link1], or on the [WorldWeatherPy][link2] page.

Thanks for reading! If you like the package, consider throwing a star on the [GitHub repository][link1], or hitting the ‘clap’ button :D

[link1]: <https://github.com/David-Woroniuk/WorldWeatherPy>
[link2]: <https://pypi.org/project/WorldWeatherPy/>
[link3]: <https://www.worldweatheronline.com/developer/>
[link4]: <https://www.worldweatheronline.com/developer/premium-api-explorer.aspx>
[link5]: <https://pypi.org/project/WorldWeatherPy/>


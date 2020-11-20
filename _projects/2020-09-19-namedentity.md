---
title: 'FOMC Named Entity Recognition — Has Information Content Evolved?'
date: 2020-09-19 13:00:00
description: Article published by Towards Data Science.
featured_image: '/images/demo/displaycy.jpg'
---

![](/images/demo/displaycy.jpg)

# FOMC Named Entity Recognition — Has Information Content Evolved?

*This article uses the “FedTools” package and SpaCy to detect and analyse Named Entities within FOMC Statements.*

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/322.embed"></iframe>

TL DR: [Github Repo][link1] and [FedTools package][link2].

## Why is interpreting FedSpeak Important?

“Fedspeak”, was initially termed by Alan Blinder to describe the “turgid dialect of English” used by Federal Reserve Chairpeople when making vague, noncommittal or ambiguous statements. Over recent years, Federal Reserve policy communications have evolved dramatically, owing to increases in the Natural Language Processing (NLP) capabilities of Financial Institutions world over. The interpretation of FOMC Statements can inform short-term trading strategies, carry trades, portfolio tilts and corporate financing strategies for Hedge Funds, Proprietary Trading Firms and Banks alike.

## So, what is Natural Language Processing?

NLP is a field of artificial intelligence enabling machines to interact with, analyse, understand and interpret the meaning of human language. NLP is comprised of a number of sub-fields, such as automated summarization, automated translation, named entity recognition, relationship extraction, speech recognition, topic segmentation and sentiment analysis.
This article focuses on implementing an information extraction technique on FOMC statements, applying named entity recognition to determine the number of times each subject is mentioned.

## Named Entity Recognition

Named entity recognition can be considered as a simple, or complex approach to information extraction, based on user preferences. A user can either use a pre-trained model, as in this tutorial, or can choose to train their own named entity recognition model.
Once the model has been trained, named entity recognition can be thought of as a combination of a detection task, and a classification task:
1) The first step involves segmenting the text into spans of tokens, otherwise known as chunking. Chunking attempts to apply non-nesting, such that ‘Federal Reserve’ is a single token, as opposed to ‘Federal’ and ‘Reserve’ existing as separate tokens.
2) The second step requires selection of an appropriate ontology by which categories are organised.

For simplicity, this tutorial uses the pre-trained ```en_core_web_sm``` model from SpaCy, which can be found [here][link3].

## Practical Implementation

Now we have an understanding of the underlying model, we can implement this in 7 easy steps.
The first step is to install the packages and modules which we shall use:

```python
# install the FedTools package:
!pip install FedTools

# install chart studio (Plotly):
!pip install chart-studio

# import pandas and numpy for data wrangling:
import pandas as pd
import numpy as np

# from FedTools, import the MonetrayPolicyCommittee module to download statements:
from FedTools import MonetaryPolicyCommittee

# import spacy and displaycy for visualisation:
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy import displacy

# import Counter for counting:
from collections import Counter

# import plotly for plotting:
import plotly.graph_objects as go
```

Secondly, we need to obtain historical FOMC statements, through the “FedTools” library, which can be found [here][link4]. Following this, additional non-text operators are removed from the dataset, and each statement is parsed, returning a DataFrame containing tokenised data, lemmatised data, part of speech tags, named entities, labels associated with the named entities,the associated number of times the labels occur, and the number of times each item is detected within the statement.

```python
def dataset_parsing():
  '''
  This function calls the MonetaryPolicyCommittee module of the FedTools package
  to collect FOMC Statements. These statements are parsed using SpaCy.
  Inputs: N/A.
  Outputs: dataset: a Pandas DataFrame which contains:
  'FOMC_Statements' - original FOMC Statements downloaded by FedTools.
  'tokenised_data' - tokenised FOMC Statements.
  'lemmatised_data' - lematised FOMC Statements.
  'part_of_speech' - part of speech tags from FOMC Statements.
  'named_entities' - the named entities detected within the FOMC Statements.
  'labels' - the corresponding labels associated with named_entities.
  'number_of_labels' - a dictionary displaying the number of each label detected.
  'items' - the number of times each item is detected within the FOMC Statements.
  '''

  # collect FOMC Statements into DataFrame called dataset:
  dataset = MonetaryPolicyCommittee().find_statements()

  # remove additional operators within the text:
  for i in range(len(dataset)):
    dataset.iloc[i,0] = dataset.iloc[i,0].replace('\\n','. ')
    dataset.iloc[i,0] = dataset.iloc[i,0].replace('\n',' ')
    dataset.iloc[i,0] = dataset.iloc[i,0].replace('\r',' ')
    dataset.iloc[i,0] = dataset.iloc[i,0].replace('\xa0',' ')

  # initialise empty lists:
  tokens = []
  lemma = []
  pos = []
  ents = []
  labels = []
  count = []
  items = []

  # for each document in the pipeline:
  for doc in nlp.pipe(dataset['FOMC_Statements'].astype('unicode').values, batch_size=50, n_threads=10):
      # if the document is successfully parsed:
      if doc.is_parsed:
          # append various data to appropriate categories:
          tokens.append([n.text for n in doc])
          lemma.append([n.lemma_ for n in doc])
          pos.append([n.pos_ for n in doc])
          ents.append([n.text for n in doc.ents])
          labels.append([n.label_ for n in doc.ents])
          count.append(Counter([n.label_ for n in doc.ents]))
          items.append(Counter([n.text for n in doc.ents]))

      # if document parsing fails, return 'None' to maintain DataFrame dimensions:
      else:
          tokens.append(None)
          lemma.append(None)
          pos.append(None)
          ents.append(None)
          labels.append(None)
          count.append(None)
          items.append(None)

  # now assign the lists columns within the dataframe:
  dataset['tokenised_data'] = tokens
  dataset['lemmatised_data'] = lemma
  dataset['part_of_speech'] = pos
  dataset['named_entities'] = ents
  dataset['labels'] = labels
  dataset['number_of_labels'] = count
  dataset['items'] = items

  return dataset
```

Now, we can begin to generate additional information from the parsed data. We search each statement’s ```number_of_labels``` column for specific label tags, identifying the number of times each of the labelled entities are mentioned. The full list of potential named entity labels can be found [here][link5].

```python
def generate_additional_information():
  '''
  This function generates additional information from the parsed documents, quantifying
  the usage of specific named entities within FOMC Statements.
  Inputs: N/A.
  Outputs: dataset: a Pandas DataFrame which contains:
  'person' - the number of times people are mentioned in each statement.
  'date' - the number of times dates are mentioned within each statement.
  'percent' - the number of times percentages are mentioned within each statement.
  'time' - the number of times a time is mentioned within each statement.
  'ordinal' - the number of times an 'ordinal' ie) "first" is mentioned within each statement.
  'organisations' - the number of times an organisation is mentioned within each statement.
  'money' - the number of times money is mentioned within each statement.
  'event' - the number of times an event is mentioned within each statement.
  'law' - the number of times a law is mentioned within each statement.
  'quantity' - the number of times a quantity is mentioned within each statement.
  'groups' - the number of times specific groups are mentioned within each statement.
  'information_content' -  the number of named entities detected within each statement.
  '''
  # call the function defined above:
  dataset = dataset_parsing()

  # generate additional information through the detection of named entities:
  dataset['person'] = dataset['number_of_labels'].apply(lambda x: x.get('PERSON'))
  dataset['date'] = dataset['number_of_labels'].apply(lambda x: x.get('DATE'))
  dataset['percent'] = dataset['number_of_labels'].apply(lambda x: x.get('PERCENT'))
  dataset['product'] = dataset['number_of_labels'].apply(lambda x: x.get('PRODUCT'))
  dataset['time'] = dataset['number_of_labels'].apply(lambda x: x.get('TIME'))
  dataset['ordinal'] = dataset['number_of_labels'].apply(lambda x: x.get('ORDINAL'))
  dataset['organisations'] = dataset['number_of_labels'].apply(lambda x: x.get('ORG'))
  dataset['money'] = dataset['number_of_labels'].apply(lambda x: x.get('MONEY'))
  dataset['event'] = dataset['number_of_labels'].apply(lambda x: x.get('EVENT'))
  dataset['law'] = dataset['number_of_labels'].apply(lambda x: x.get('LAW'))
  dataset['quantity'] = dataset['number_of_labels'].apply(lambda x: x.get('QUANTITY'))
  dataset['groups'] = dataset['number_of_labels'].apply(lambda x: x.get('NORP'))

  # replace any 'NaN' values with 0, then calculate the 'information content',as defined
  # by the total number of named entities:
  dataset = dataset.replace(np.nan, 0)
  dataset['information_content'] = dataset.iloc[:,8:].sum(axis = 1)

  return dataset
```

Next, we can try to detect the relevant chairperson within each FOMC statement. The can be used as a quick identification tool for the incumbent Chair of the FOMC, although may generate a ‘NaN’ value if the board member’s name is not present within the statement (as in earlier years within the sample).

```python
def generate_chairperson(dataset):
  '''
  This function uses Named Entity Recognition in order to detect the presence of 
  chairpeople within the FOMC statements. 
  Inputs: dataset: a Pandas DataFrame as defined above.
  Outputs: dataset: a Pandas DataFrame which identifies the FOMC Chairperson.
  '''

  # try to detect specific names within 'items':
  dataset['Greenspan'] = dataset['items'].apply(lambda x: x.get('Alan Greenspan'))
  dataset['Bernanke'] = dataset['items'].apply(lambda x: x.get('Ben S. Bernanke'))
  dataset['Yellen'] = dataset['items'].apply(lambda x: x.get('Janet L. Yellen'))
  dataset['Powell'] = dataset['items'].apply(lambda x: x.get('Jerome H. Powell'))

  # replace all 'Nan' values with 0:
  dataset = dataset.replace(np.nan, 0)

  return dataset
```

We can now generate an interactive Plotly chart which outlines the total information content detected within each statement, and decomposes this into specific labelled entities. This is particularly useful visually inspecting potential Hypotheses, such as: “FOMC Statements prior to US Election cycles contain little information” or “Statements made during market crises contain more quantitative information”.

```python
def plot_figure():
  '''
  This function constructs a Plotly chart by calling the above functions to generate
  the dataset, and subsequently plotting relevant data. 
  '''

  # define the dataset as a global variable, which can be used outside of the function:
  global dataset
  # call the above functions to generate the required data:
  dataset = generate_additional_information()
  dataset = generate_chairperson(dataset)

  # initialise figure:
  fig = go.Figure()

  # add figure traces:
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['information_content'],
                           mode = 'lines',
                           name = 'Information Content',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['percent'],
                           mode = 'lines',
                           name = 'Number of times "Percentage" mentioned',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['person'],
                           mode = 'lines',
                           name = 'Number of People mentioned',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['money'],
                           mode = 'lines',
                           name = 'Number of times Money mentioned',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['quantity'],
                           mode = 'lines',
                           name = 'Number of Quantities mentioned',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['event'],
                           mode = 'lines',
                           name = 'Number of Events mentioned',
                           connectgaps=True))
  
  fig.add_trace(go.Scatter(x = dataset.index, y = dataset['organisations'],
                           mode = 'lines',
                           name = 'Number of Organisations mentioned',
                           connectgaps=True))

  # add a rangeslider and buttons:
  fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=5, label="5 Years", step="year", stepmode="backward"),
            dict(count=10, label="10 Years", step="year", stepmode="backward"),
            dict(count=15, label="15 Years", step="year", stepmode="backward"),
            dict(label="All", step="all")
        ]))) 

  # add a chart title and axis title:
  fig.update_layout(
    title="FOMC Named Entity Recognition",
    xaxis_title="Date",
    yaxis_title="",
    font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
    ))
  
  # add toggle buttons for dataset display:
  fig.update_layout(
      updatemenus=[
          dict(
            buttons=list([
                  dict(
                    label = 'All',
                    method = 'update',
                    args = [{'visible': [True, True, True, True, True, True, True]}]
                  ),

                  dict(
                    label = 'Information Content',
                    method = 'update',
                    args = [{'visible': [True, False, False, False, False, False, False]}]
                  ),

                  dict(
                    label = 'Percentage mentions',
                    method = 'update',
                    args = [{'visible': [False, True, False, False, False, False, False,]}]
                  ),

                  dict(
                    label = 'People mentions',
                    method = 'update',
                    args = [{'visible': [False, False, True, False, False, False, False,]}]
                  ),

                  dict(
                    label = 'Money mentions',
                    method = 'update',
                    args = [{'visible': [False, False, False, True, False, False, False,]}]
                  ),

                  dict(
                    label = 'Quantity mentions',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, True, False, False,]}]
                  ),

                  dict(
                    label = 'Event mentions',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, False, True, False,]}]
                  ),

                  dict(
                    label = 'Organisation mentions',
                    method = 'update',
                    args = [{'visible': [False, False, False, False, False, False, True]}]
                  ),
              ]),
              direction="down",
              pad={"r": 10, "t": 10},
              showactive=True,
              x=1.0,
              xanchor="right",
              y=1.2,
              yanchor="top"
          ),])
  
  return fig.show()
```

Now, we can call the ```plot_figure function```, which internally calls the above functions to generate and parse the data, finally plotting the number of named entities detected.

```python
plot_figure()
```

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/~DavidWoroniuk/322.embed"></iframe>

Additionally, ```displacy.render``` can be called on individual FOMC Statements, enabling visual analysis of the entity recognition accuracy.
```python
displacy.render(nlp(dataset['FOMC_Statements'][103]), jupyter = True, style = 'ent')
```

To select which FOMC statement you would like to visualise, change the numerical value within the ```(dataset[‘FOMC_Statements’][103])``` to a different value. ```-1``` (negative indexing) provides a visualization of the most recent statement.



[link1]: <https://github.com/David-Woroniuk/Medium-Articles/blob/master/Named_Entity_Recognition.ipynb>
[link2]: <https://pypi.org/project/FedTools/>
[link3]: <https://spacy.io/models/en>
[link4]: <https://pypi.org/project/FedTools/>
[link5]: <https://spacy.io/api/annotation#named-entities>



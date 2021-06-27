---
date: 2021-03-15
title: "Project: Stock level prediction"
linkTitle: Stocks
tags: ["project", "ai", "finance"]
description: "This project includes a deep learning model for stock prediction. It uses LSTM, RNN which is the standart for time series prediction. It seems to be the right approach. The author really loved this project since he loves stocks. 
He invests often, and is also in love with tech, so he finds ways to combine both of them. Most existing models for stock prediction do not include the volume, and Rishabh intendede to use that as an input, but it did not go exactly as planned."
author: Rishabh Agrawal
github_url: https://github.com/cybertraining-dsc/sp21-599-353/edit/main/project/index.md
resources:
- src: "**.{png,jpg}"
  title: "Image #:counter"
---


[![Check Report](https://github.com/cybertraining-dsc/sp21-599-353/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-353/actions)
[![Status](https://github.com/cybertraining-dsc/sp21-599-353/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/sp21-599-353/actions)
Status: final, Type: Project

Rishabh Agrawal, [sp21-599-353](https://github.com/cybertraining-dsc/sp21-599-353/), [Edit](https://github.com/cybertraining-dsc/sp21-599-353/blob/main/project/index.md)
* [Stock Prediction.pynb](https://github.com/cybertraining-dsc/sp21-599-353/blob/main/project/code/Stock%20Prediction.ipynb)
* [Stock Prediction.pdf](https://github.com/cybertraining-dsc/sp21-599-353/blob/main/project/code/Stock%20Prediction.pdf)

{{% pageinfo %}}

## Abstract

This project includes a deep learning model for stock prediction. It uses LSTM, RNN which is the standart for time series prediction.
It seems to be the right approach. The author really loved this project since he loves stocks. 
He invests often, and is also in love with tech, so he finds ways to combine both of them. Most existing models for stock prediction 
dont include the volume, and Rishabh intendede to use that as an input, but it didn't go exactly as planned.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}


**Keywords:** tensorflow, LSTM, Time Series prediction, transformers.

## 1. Introduction

Using deep learning for stock level prediction is not a new concept, but this project is trying to address a different issue her. Volume. Almost no model actually uses volume, daily volume or weekly volume.
People that are experts in this field and do a technical analysis(TA) of stocks use volume for their prediction extensively, so why not use it in the model? It could be ground breaking. 

LSTM is the obvious match for this kind of problem, but this project will also try to incorporate transformers in the model. The data set used will be from Yahoo Finance[^1]. 


## 2. Existing LSTM Models

There are quite a few models out there for stock prediction. But what exactly is LSTM? How does it work, and how is it the obvious option here? There are other models too. 
Why the current LSTM models aren't that great? How good/bad do they perform?

## 2.1. What is LSTM?

LSTM is short for Long Short Term Memory networks. It comes under the branch of Recurrant Nueral Networks. LSTMs are explicitly designed to avoid the long-term dependency problem. 
Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer[^2].
LSTM is also used for other time series forecasting such as weather and climate. It is an area of deep learning that works on considering the last few instances of the time series instead of the entire time series as a whole. 
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor[^3].

![LSTM](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/lstm.png)
<br />Fig 1 LSTM

This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They’re the natural architecture of neural network to use for such data. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn.

## 2.2. Existing LSTM models and why they don't do great

This data corroborates what we can see from Figure 4. The low values in RMSE and decent values in R2 show that the LSTM may be good at predicting the next values for the time series in consideration.

Figure 5 shows a sample of 100 actual prices compared to predicted ones, from August 13, 2018 to January 4, 2019.
[![LSTM prediction](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/lstm_pred.png)](https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/)
<br />Fig 5 LSTM Prediction 1 [Image Source](https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/)

This figure makes us draw a different conclusion. While in aggregate it seemed that the LSTM is effective at predicting the next day values, 
in reality the prediction made for the next day is very close to the actual value of the previous day. 
This can be further seen by Figure 6, which shows the actual prices lagged by 1 day compared to the predicted price[^4].

[![LSTM prediction2](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/lstm_pred2.png)](https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/)
<br />Fig 6 LSTM Prediction 2 [Image Source](https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/)

In one other model on Google Colab we see an output like this

[![LSTM prediction3](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/lstm_pred3.png)](https://colab.research.google.com/github/Mishaall/Geodemographic-Segmentation-ANN/blob/master/Google_Stock_Price_Prediction_RNN.ipynb#scrollTo=skhdvmCywHrr)
<br />Fig 7 LSTM Prediction 3 [Image Source](https://colab.research.google.com/github/Mishaall/Geodemographic-Segmentation-ANN/blob/master/Google_Stock_Price_Prediction_RNN.ipynb#scrollTo=skhdvmCywHrr)

Here we can see that clearly the model did not do well. In this model, the LSTM didn't even get the trends correctly. There is definitely need for some changes in the layers, and droupout coeffecient. 
My guess is that this model was really overfitted since the first half of the prediction did reletively well[^5].

## 3. Datasets

As mentioned above, the Yahoo finance Data set will be used. It is really easy to get. Data can be download from any time stamp to as new as today. There is a download csv button to do so. 
Here is an example of [AAPL stock](https://finance.yahoo.com/quote/aapl/history?ltr=1) on Yahoo Finance[^1].

For this project the Amazon stock was chosen. The all time max historic data was downloaded from Yahoo Finance in csv format. Here is the [link](https://github.com/cybertraining-dsc/sp21-599-353/blob/main/project/code/AMZN_Stock_Price.csv)

## 4. Results

Fig 8 shows the result of my model. The results weren't as expected. This project was meant to be for adding volume to the input layer. We tried to do that in the model, but it failed. It gave a straight 
line for the prediction. This project did find a unique way to use the LSTM. It played around with the layers, number of layers, the droupout coeffecient to find the most accurate balance. The output shows a more 
reliable and believable output, and that is some progress. There might be ways to incorporate volume. Later, maybe it needs better scaling. But the prediction did get the trends pretty accurately, even
though it might not have gotten the exact price correctly. Amazon stock alse soared unbelievabley this year due to COVID-19 and many other external factors that were not incorporated in the model at 
all, so it is really common to see an undervalued prediction.<br />
![LSTM results](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/AMZN_stock_prediction_graph.png)
<br />Fig 8

## 6. Benchmark

Cloudmesh was used for the benchmark for this project. According to the documentaion[^6], I used the StopWatch.start() and StopWatch.stop() functions. Fig 9 shows the output. Loading the dataset or prediction of the data doesn't take long at all. The majority of the time is taken for the training 
wihch is expected. Google Colab was the tool used and utilized the GPU which is why each epoch just took about 2 seconds. When a personal CPU or the default CPU provided by Google Colab was used, 
it took about 30-40 seconds for each epoch. You can see the system configuration in Fig 9 too.
![Cloudmesh Benchmark](https://github.com/cybertraining-dsc/sp21-599-353/raw/main/project/images/benchmark.png)
<br />Fig 9 Cloudmesh Benchmark Results
 
## 7. Conclusion

Even though the model did not do as expected, we were not able to add volume as an input, we were still able to find some success with changing the number layers and the coeffecient values. We can see
that the model can successfully predict the trend of the stock prices, even with the external factors affecting the prices a little. The next step for this project would be to try to scale the volume
and add it as an input to the model. One other, but rather difficult add-on could be to try to add some external factors as inputs. For an aggregate input of external factors, we could use 
sentiment analysis through Twitter tweets as an input to the model too.

## 9. References

Your report must include at least 6 references. Please use customary academic citation and not just URLs. As we will at 
one point automatically change the references from superscript to square brackets it is best to introduce a space before 
the first square bracket.

[^1]: Using dataset for stocks, [Online resource] 
      <https://finance.yahoo.com/>

[^2]: Understanding LSTM Networks, [Online Resource] 
      <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>

[^3]: AI stock market forecast, [Online resource]
      <http://ai-marketers.com/ai-stock-market-forecast/>

[^4]: Why You Should Not Use LSTM’s to Predict the Stock Market, [Online resource]
      <https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/>

[^5]: Google Stock Price Prediction RNN, [Google Colab]
      <https://colab.research.google.com/github/Mishaall/Geodemographic-Segmentation-ANN/blob/master/Google_Stock_Price_Prediction_RNN.ipynb#scrollTo=skhdvmCywHrr>

[^6]: Cloudmesh, [Documentation]
      <https://cloudmesh.github.io/cloudmesh-manual/autoapi/cloudmeshcommon/cloudmesh/common/StopWatch/index.html#module-cloudmesh-common.cloudmesh.common.StopWatch>

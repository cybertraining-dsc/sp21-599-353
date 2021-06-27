# HW 6
Author: Rishabh Agrawal

I am going to continue with what I had with stock level prediction in homework 3. 

With stocks, almost all the date is pretty much open source. I can use yahoo finance to get the data.
For example if I want Apple’s stock price, I can use this website to download the excel for the open, close, high, low and volume for a day for the past how many ever days I would like. Ideally I am looking for about 5-6 years since I will have more data to work with and the model will be more accurate. (https://finance.yahoo.com/quote/aapl/history?ltr=1)[https://finance.yahoo.com/quote/aapl/history?ltr=1]


The paper I found on this was this: (https://arxiv.org/pdf/2004.10178v1.pdf)[https://arxiv.org/pdf/2004.10178v1.pdf]
They use LSTM and random forest to make the prediction, which is said to be one of the best for any time series data. They also used different error checking methods to see how accurate they were. It wasn’t that accurate, otherwise it would have been broken, but it was pretty accurate for a lot of them. I also managed to find the GitHub code for it. They used Tensor flow for that. They had a intraday and a next day trading code. In the results section they compared how it was compared to the actual values in a bar and a line graph which a thought was really useful (https://github.com/pushpendughosh/Stock-market-forecastings)[https://github.com/pushpendughosh/Stock-market-forecastings]

# Capston Project  <!-- omit in toc -->

Machine Learning Engineer Nanodegree

Chen Tong 
Jun 27, 2019

Table of content
- [Definition](#Definition)
  - [Project Overview](#Project-Overview)
  - [Problem Statement](#Problem-Statement)
  - [Metrics](#Metrics)
- [Analysis](#Analysis)
  - [Data Exploration and Visualization](#Data-Exploration-and-Visualization)
    - [Portfolio Data](#Portfolio-Data)
    - [Profile Data](#Profile-Data)
    - [Transcript Data](#Transcript-Data)
  - [Algorithms and Techniques](#Algorithms-and-Techniques)
  - [Benchmark](#Benchmark)
- [Methodology](#Methodology)
  - [Data Preprocessing](#Data-Preprocessing)
  - [Implementation](#Implementation)
  - [Refinement](#Refinement)
- [Results](#Results)
  - [Model Evaluation and Validation](#Model-Evaluation-and-Validation)
  - [Justification](#Justification)
- [Conclusion](#Conclusion)


## Definition

### Project Overview
Starbucks is an American coffeehouse chain. Once every few days, Starbucks sends out an offer to users via different ways such as mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. An important characteristic regarding this capstone is that not all users receive the same offer. As part of marketing strategy, we always want to figure out if a customer will spend more by giving a sound offer. Providing right offer to right customer could help build loyalty of the brand and product and as a result increasing sales margins in the long run.  
The goal is to create a predictor to answer a question, that is, if a customer will response and complete a given offer? 

The data is contained in three files. The files are in `data` folder in this repo:

- portfolio.json: offer ids and meta data about each offer (duration, type, etc.)
- profile.json: demographic data for each customer
- transcript.json: records for transactions, offers received, offers viewed, and offers completed

These files are cleaned, processed, transformed and joined into one table which contains demographic data of a customer, attributes of the offer and whether the customer complete the offer. The final dataset is 76277 * 20 in shape and columns or features are age, became_member_on, income, gender_F, gender_M, gender_O, gender_nan, difficulty, duration, reward, offer_type_bogo, offer_type_discount, offer_type_informational, channel_email, channel_web, channel_mobile, channel_social, reward%difficulty, difficulty%duration.  

Labels values are balanced because the number of occurrences of positive label is 34809 while the number is 41468.

Below are the first 5 rows of the dataset. The first column is label. The rest are features.  
![Sample Data 1](images/sample1.png)
![Sample Data 2](images/sample2.png)


### Problem Statement

I chose to build a model that predicts whether or not a customer will complete an offer, meaning that the customer will receive a offer, view the offer and finish the offer before expire day. Thus, to solve this problem, I will establish a binary classifier.  

One potential solution has 3 steps:
1. Preprocess combined transaction, demographic and offer data. This dataset describes an offer’s attributes, the user's demographic data and whether the offer was successful. Also, split datasets into train and test datasets.    
2. Training XGBoost Model as benchmark model.  
3. Training PyTorch deep learning model and improve the results.  

### Metrics

The evaluation metric is the Receiver Operating Characteristic Curve (ROC AUC) score from prediction scores. It computes area under the Receiver Operating Characteristic Curve. This will quantify the performance of both the benchmark model and the solution model. The reason is that the datasets are balanced and also, we only care about the final class predictions and we don’t want to tune threshold.  

## Analysis

### Data Exploration and Visualization

#### Portfolio Data
This dataset is about offers sent during 30-day test period. There are totally 10 offers. 4 of them are BOGO, 4 are discount and 2 are informational. All offer are distributed via email. By contrast only 6 offers are showed in social networks. 

#### Profile Data
This dataset describes rewards program users, 17000 users totally. 

We show customers' age in the interval of 5 years. The missing value is encoded as 118, which is more than 2000 data points. Most of the customers fall in the age range from 50 to 60 years old.  
![Person Age Distribution](images/persons_by_age.png)

Most of the customers fall in the income range between 50000 and 60000. The income range from 40000 to 120000. 
![Person Income Distribution](images/persons_by_income.png)

As for gender, there are two other categories than female and male. Male customers are more than female customers and gener O is smallest in number.  
![Person Gener Distribution](images/persons_by_gender.png)

Member grows by year and get to the peak at 2017. The plot suggest that most customers recently joined the rewards program. 
![Person Member Distribution](images/persons_by_member.png)


#### Transcript Data  
This dataset contains event log. There are 306648 events. Similarly, we will explore insights of transactions.  

There are four events, offer received, offer viewed, transaction, offer completed. We could see the amount of completed offer is around half of that of offer received. Offer viewed is a bit lower than offer received. It hints the dataset would not be perfectly balanced.  
![Transcript Event](images/transcript_by_event.png)

When discovering this datasets, we do see some abnormalities. For example, offer type informational usually don't have a complete event. See below summary for one person. 

![offer type informational](images/transcript_offer_type2.png)

The second row shows after viewing this offer, the user spend 49.39 dollars. We think such behavior should be consider a success offer, thus we adjust the class to completed offer.  
We also see behavior like no money spent after viewing informational offer. We don't category this to completed offer. Example provieded following:

![information no money](images/informational_zero.png)

According to the instruction of this project, we should not consider offer complete if people didn't view it. See below to find this case.  

![complete without view](images/complete_no_view.png)

We could clearly see the offer id `290` is received and completed without view event. However, we see interesting behavior like offer viewed after offer completed. See below records:

![complete before view](images/complete_then_view.png)

The offer is viewed after completion. We also observe that the time offer viewed is still in valid duration and amount after offer received exceed difficulty (required amount). We consider this case also completed offer because customer may see the offer but didn't take any response to it. But we don't consider the offer is completed if any requirements such as duration or difficulty are satisfied. 

### Algorithms and Techniques

### Benchmark

## Methodology

### Data Preprocessing

`offer_type` and `channels` are category data. We could do one-hot encoding on these two columns. The data

### Implementation

### Refinement

## Results

### Model Evaluation and Validation

### Justification


## Conclusion

Improvement

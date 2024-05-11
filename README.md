# MovieAnalyzer

## Abstract

In today's world, the enormous volume of movie reviews scattered across various platforms poses a significant challenge in comprehensively understanding public sentiments. To tackle this challenge, we introduce an automated approach employing sentiment analysis (SA) and natural language processing (NLP) techniques to classify sentiments within a dataset comprising 10,000 IMDb movie reviews. Leveraging logistic regression, our methodology achieves an impressive accuracy of 0.88, recall of 0.9, and F1-score of 0.88. This automated sentiment classification facilitates the efficient extraction of public sentiment, thereby offering valuable insights for individuals seeking informed decision-making in movie selection.

## Table of Contents
1. [Discover Phase Problem](#discover-phase-problem)
2. [Data Preparation Phase](#data-preparation-phase)
3. [Model Planning Phase](#model-planning-phase)
4. [Model Building Phase](#model-building-phase)
5. [Communication Phase](#communication-phase)
6. [Conclusion](#conclusion)

## 1. Discover Phase Problem

The proliferation of the film industry has led to an abundance of opinions expressed about movies across various platforms. However, navigating through online movie reviews to gauge public sentiment is daunting due to the sheer volume of data. IMDb, a prominent platform for aggregating movie reviews, amassed a staggering 7 million reviews by 2023. This abundance of data makes it arduous for viewers to ascertain whether a film is well-received or not.

**Goal:** Our objective is to utilize SA and NLP techniques to efficiently process large volumes of movie reviews, empowering individuals to make informed decisions about which movies to watch.

## 2. Data Preparation Phase

### Dataset Description

The dataset, obtained from Kaggle, consists of 50,000 movie reviews equally distributed between positive and negative sentiments. However, due to computational limitations, we downscaled the dataset to 10,000 reviews, evenly split between positive and negative sentiments.

### Data Cleaning

Raw text often contains noise, biases, and inconsistencies that can impede model performance. Hence, data cleaning is crucial to refine and preprocess the text, eliminating biases, noise, and ensuring consistency. We utilized the `tm` package for data mining and cleaning, along with the `textstem` package for lemmatization. Our cleaning process involved:
- Converting text to lowercase for uniformity.
- Removing whitespace, punctuation, numbers, stop words, URLs, hashtags, and redundant whitespaces.
- Lemmatization to reduce words to their base or root form.

## 3. Model Planning Phase

### Visualization

#### Bag of Words (BOW)

We utilized the DocumentTermMatrix function to compute the word frequency in each review, converting the reviews into columns and rows.

#### Barplots

Bar plots were employed to visually present the top 30 positive and negative words from the dataset.

#### Wordclouds

Word clouds were used to showcase the top 200 positive and negative words, with word size indicating frequency or significance.

### Model Selection

Logistic regression, a supervised machine learning algorithm, was chosen for binary classification tasks in SA due to its simplicity and effectiveness with text data.

## 4. Model Building Phase

### Logistic Regression in SA

Logistic regression emerged as an optimal choice for SA due to its simplicity and interpretability, effectively handling complexity even with large text datasets.

### Model Building Steps

- **Splitting the data:** The data was split into training and testing sets using a hold-out technique.
- **Building the model:** Parallel processing was employed for efficient model training, utilizing the `cv.glmnet` function for regularization and handling high-dimensional features.
- **Evaluating the model:** Model performance was evaluated using accuracy, recall, and F1-score metrics.

## 5. Communication Phase

### Study Outcomes

We achieved an accuracy of 0.88, recall of 0.9, and F1-score of 0.88, showcasing the effectiveness of our approach.

### Encountered Problems

Computational limitations necessitated downsizing the dataset to 10,000 reviews and implementing optimization techniques like `cv.glmnet` and parallel processing.

## 6. Conclusion

The MovieAnalyzer project aims to aid individuals in making informed decisions about movie quality based on reviews. Leveraging SA and NLP techniques in R, our model accurately predicts sentiment with an accuracy of 0.88. Future research could delve into finer-grained SA and domain adaptation techniques to enhance model accuracy across diverse movie datasets. Additionally, exploring alternative approaches such as Naive Bayes, Support Vector Machines, or Transformer-based Models could offer further insights and improvements.

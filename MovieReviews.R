## --------------------- ##
## DOWNLOADING LIBRARIES ##
## --------------------- ##

install.packages(c('caTools', 'glmnet', 'doParallel', 'Matrix', 'Metrics', 'MLmetrics'
                   ,'textstem', 'dplyr', 'ggplot2', 'wordcloud2', 'tm' , 'tidytext' , 'textdata'))

## ----------------- ##
## EXAMINING DATASET ##
## ----------------- ##

# Examine dataset
head(Movie_Dataset)
tail(Movie_Dataset)
summary(Movie_Dataset)

# Number of positive and negative reviews
table(Movie_Dataset$sentiment)

# Our computers can't handle 50,000 rows. Therefore, the number of rows
# is reduced to 10,000, with equal number of positive and negative reviews.

library(dplyr)
set.seed(123)  # for reproducibility, ensuring that the same split is happened each time running the code

# Separate positive and negative reviews
positive_reviews <- Movie_Dataset %>% filter(sentiment == "positive")
negative_reviews <- Movie_Dataset %>% filter(sentiment == "negative")

# Randomly sample 5,000 positive and 5,000 negative reviews
sampled_positive <- positive_reviews %>% sample_n(5000)
sampled_negative <- negative_reviews %>% sample_n(5000)

# Bind the sampled reviews
Movie_Dataset <- bind_rows(sampled_positive, sampled_negative)

# Number of new positive and negative reviews
table(Movie_Dataset$sentiment)


## ------------------------------- ##
## NULL VALUES ADDING AND EXLUDING ##
## ------------------------------- ##

# Check for null values
sum(is.na(Movie_Dataset))

# Adding null row into the dataset
new_null_row <- data.frame(review = NA, sentiment = NA)
Movie_Dataset <- rbind(Movie_Dataset, new_null_row)

# Checking number for NA values, resulting in 2 column with NA
sum(is.na(Movie_Dataset))

# Removing null values
Movie_Dataset <- na.exclude(Movie_Dataset)
sum(is.na(Movie_Dataset))


## --------------##
## DATA CLEANING ##
## --------------##

library(tm) # Text mining and data cleaning
library(textstem) # Lemmitization 

# Creating a Corpus "collection of text documents, required in tm package to make pre-processing"
Movie_Corpus <-Corpus(VectorSource(Movie_Dataset$review))

Movie_Corpus <- tm_map(Movie_Corpus, content_transformer(tolower)) #lowercase
Movie_Corpus <- tm_map(Movie_Corpus, stripWhitespace) # removes whitespaces
Movie_Corpus <- tm_map(Movie_Corpus, removePunctuation) # removes punctuations
Movie_Corpus <- tm_map(Movie_Corpus, removeNumbers) # removes numbers
Movie_Corpus<- tm_map(Movie_Corpus, removeWords, stopwords("english")) #remove stopwords

# Custom function to remove URLs, hashtags, controls, and extra whitespaces
text_preprocessing <- function(x) {
  gsub('http\\S+\\s*|#\\S+|[[:cntrl:]]','',x)
  gsub("^[[:space:]]*|[[:space:]]*$| +", " ", x)
}

# Applying the function
Movie_Corpus <-tm_map(Movie_Corpus,text_preprocessing)

# Lemmatization
Movie_Corpus <- tm_map(Movie_Corpus, textstem::lemmatize_words)
Movie_Corpus <- tm_map(Movie_Corpus, content_transformer(lemmatize_strings))

# Binding cleaned data with the original dataset
content <- content(Movie_Corpus)
Movie_Dataset <- cbind(Movie_Dataset, LargeChar = content)

# Changing the name of cleaned column
colnames(Movie_Dataset)[3] <- "cleaned_review"


## -------------------##
## HYPOTHESIS TESTING ##
## -------------------##

library(tidytext)
library(textdata)

sentiment_lexicon <- get_sentiments("afinn") #The AFINN lexicon is a list of words with associated sentiment scores.
reviews <- Movie_Dataset %>%
  unnest_tokens(word, review)

reviews_sentiment <- inner_join(reviews, sentiment_lexicon, by = "word")

positive_sentiment <- subset(reviews_sentiment, sentiment == "positive")
negative_sentiment <- subset(reviews_sentiment, sentiment == "negative")

ttest_result <- t.test(positive_sentiment$value, negative_sentiment$value)
ttest_result

# p-value < 2.2e-16  
# Therefore, we reject null hypothesis and accept alternative hypothesis
# which indicates that there a significant difference in sentiment between positive and negative reviews.


## -------------------##
## DATA VISUALIZATION ##
## -------------------##

# Convert the text data into a BOW representation
dtm <- DocumentTermMatrix(Movie_Corpus)

# Subset the bag of words matrix based on sentiment
positive_words <- dtm[Movie_Dataset$sentiment == "positive", ]
negative_words <- dtm[Movie_Dataset$sentiment == "negative", ]

# Calculate word frequencies for positive and negative words
positive_freq <- colSums(as.matrix(positive_words))
negative_freq <- colSums(as.matrix(negative_words))

# Sort the word frequencies in descending order
sorted_positive_freq <- sort(positive_freq, decreasing = TRUE)
sorted_negative_freq <- sort(negative_freq, decreasing = TRUE)

# Select the top 30 words. The first two word are excluded (Movie and Film)
top_positive_words_30 <- names(sorted_positive_freq)[3:33]
top_negative_words_30 <- names(sorted_negative_freq)[3:33]

# Create a data frame for the top positivea and negative words and their frequencies
positive_df <- data.frame(Word = top_positive_words_30, Frequency = sorted_positive_freq[top_positive_words_30])
negative_df <- data.frame(Word = top_negative_words_30, Frequency = sorted_negative_freq[top_negative_words_30])


library(ggplot2)

ggplot(positive_df, aes(x = Frequency, y = reorder(Word, -Frequency), fill = Word)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 30 Positive Word Frequencies",
       x = "Word",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")

ggplot(negative_df, aes(x = Frequency, y = reorder(Word, -Frequency), fill = Word)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 30 Negative Word Frequencies",
       x = "Frequency",
       y = "Word") +
  theme_minimal() +
  theme(legend.position = "none")


library(wordcloud2)

# Select the top 200 words. The first two word are excluded (Movie and Film)
top_200_positive_words <- names(sorted_positive_freq)[3:202]
top_200_negative_words <- names(sorted_negative_freq)[3:202]

# Create a data frame for the top positive words and their frequencies
positive_df_200 <- data.frame(Word = top_200_positive_words, Frequency = sorted_positive_freq[top_200_positive_words])

# Create a data frame for the top negative words and their frequencies
negative_df_200 <- data.frame(Word = top_200_negative_words, Frequency = sorted_negative_freq[top_200_negative_words])

wordcloud2(positive_df_200, size = 0.8)
wordcloud2(negative_df_200, size = 0.8)


## ----------------------------------------##
## APPLYING LOGISTIC REGRESSION ALGORITHM  ##
## ----------------------------------------##

library(caTools)    # for sample split 
library(glmnet)     # for logistic regression
library(doParallel) # for parallel processing 
library(Matrix)     # for sparse matrix support
library(Metrics)    # for accuracy, recall
library(MLmetrics)  # for F1_Score

# 1. Split data into training and testing datasets (80% training, 20% testing)
set.seed(123)  # for reproducibility, ensuring that the same split is happened each time running the code
split <- sample.split(Movie_Dataset$sentiment, SplitRatio = 0.8)
train_data <- subset(Movie_Dataset, split == TRUE)
test_data <- subset(Movie_Dataset, split == FALSE)

# Convert sentiment to 0 and 1
Movie_Dataset$sentiment <- as.numeric(Movie_Dataset$sentiment == "positive")

# Convert text data to a sparse matrix
corpus <- Corpus(VectorSource(Movie_Dataset$cleaned_review))
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
sparse_dtm <- as(dtm, "sparseMatrix")

# 2. Train the logistic regression model using glmnet with parallelization
cl <- makeCluster(detectCores() - 1)  # using one less core to avoid overloading
registerDoParallel(cl)

model <- cv.glmnet(sparse_dtm, Movie_Dataset$sentiment, family = "binomial", alpha = 1, type.measure = "class")

stopCluster(cl)  # stop the parallel backend

# 3. Evaluate the model on the testing set
predictions <- predict(model, newx = sparse_dtm, type = "response")
threshold <- 0.5  
predicted_labels <- ifelse(predictions > threshold, 1, 0)

# Calculate accuracy, recall, and F-score
accuracy_value <- accuracy(Movie_Dataset$sentiment, predicted_labels)
recall_value <- recall(Movie_Dataset$sentiment, predicted_labels)
f_score_value <- F1_Score(Movie_Dataset$sentiment, predicted_labels)

cat("Accuracy:", accuracy_value, "\n")
cat("Recall:", recall_value, "\n")
cat("F-score:", f_score_value, "\n")

# OPAN 6603 - Project 2 ####

# Mike Johnson | Andrew Singh

## Set up ####

# Load Libraries
library(tidyverse)
library(GGally) # for data exploration
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(pROC)  #For ROC curve and estimating the area under the ROC curve
library(fastDummies) #To create dummy variable (one hot encoding)

# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

## Load Data ####
df = read.csv('AirbnbListings.csv')

# Check data structure
str(df)

# Update data types
df = 
  df %>% 
  mutate(
    neighborhood = factor(neighborhood),
    host_since = as.Date(host_since, format = "%m/%d/%Y"),
    room_type = factor(room_type),
    bathrooms = as.numeric(sub(" .*", "", df$bathrooms)) # Strip out "bath" and make numerical.
  )

# Remove any irrevelant columns
df = 
  df %>% 
  select(!listing_id)

# Create dummy variables for categorical variables
df = 
  df %>% dummy_cols(select_columns = c('neighborhood', 'room_type'),
                    remove_selected_columns = T,
                    remove_first_dummy = F)



## Step 1: Create a train/test split ####

# Divide 30% of data to test set
test_indices = createDataPartition(1:nrow(df),
                                   times = 1,
                                   p = 0.3)

# Create training set
df_train = df[-test_indices[[1]], ]

# Create test set
df_test = df[test_indices[[1]], ]

## Step 2: Data Exploration ####

# Summary of training set
summary(df_train)

### host_since vs price ####
# There doesn't appear to be any relationship between price and host_since. Consider removing.
df_train %>% 
  ggplot(aes(x = host_since, y = price)) +
  geom_point(color = "steelblue") +
  labs(title = "Host Since vs Price")

### superhost vs price ####
# There doesn't appear to be any relationship between price and superhost. Consider removing.
df_train %>% 
  ggplot(aes(x = superhost, y = price)) +
  geom_boxplot(color = "steelblue") +
  labs(title = "Superhost vs Price")

### host_acceptance_rate vs price ####
# Data clustered around 1.00.
df_train %>% 
  ggplot(aes(x = host_acceptance_rate, y = price)) +
  geom_point(color = "steelblue") +
  labs(title = "Host Since vs Price")

### accommodates vs price ####
# A lot of listings accommodate between 2 - 4. Listings that accommodate > 6 are sporadic.
# The more a listing accommodates means a higher price on average.
df_train %>% 
  ggplot(aes(x = factor(accommodates), y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Accommodates vs Price",
       x = "Accomodates",
       y = "Price")

### bathrooms vs price ####
# A lot of listings have only 1 bathroom. Listings with > 2.5 bathrooms are sporadic.
# The more bathroom means a higher price on average.
df_train %>% 
  ggplot(aes(x = factor(bathrooms), y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Bathrooms vs Price",
       x = "Bathrooms",
       y = "Price")

### bedrooms vs price ####
# A lot of listings between 1-2 bedrooms. Listings > 3 bedrooms are sporadic.
# Need NA handling.
df_train %>% 
  ggplot(aes(x = factor(bedrooms), y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Bedrooms vs Price",
       x = "Bedrooms",
       y = "Price")

### bed vs price ####
# A lot of listings between 1-2 beds. Listings > 3 bedrooms are sporadic.
# Consider using beds to inpute bedrooms NA's...
df_train %>% 
  ggplot(aes(x = factor(beds), y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Beds vs Price",
       x = "Beds",
       y = "Price")

### min_nights vs price ####
# Long-term stays are very rare...
df_train %>% 
  ggplot(aes(x = min_nights, y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Min Nights vs Price",
       x = "Min Nights",
       y = "Price")

# Let's zoom in to see what's going on from 0-100 Days...
# Two observable clusters around 0 days and ~30 days.
df_train %>% 
  ggplot(aes(x = min_nights, y = price)) +
  geom_jitter(color = "steelblue") +
  labs(title = "Min Nights vs Price",
       x = "Min Nights",
       y = "Price") + 
  xlim(0, 100)

### total_reviews vs price ####
# No noticeable relationship between the two variables. Consider removal.
df_train %>% 
  ggplot(aes(x = total_reviews, y = price)) +
  geom_point(color = "steelblue") +
  labs(title = "Total Reviews vs Price",
       x = "Total Reviews",
       y = "Price")

### avg_rating vs price ####
# Clustered around 4-5. No particular pattern. Consider removal.
df_train %>% 
  ggplot(aes(x = avg_rating, y = price)) +
  geom_point(color = "steelblue") +
  labs(title = "Avg Rating vs Price",
       x = "Avg Rating",
       y = "Price")

### bed vs bedroom ####
# Consider using beds to inpute bedrooms.
df_train %>% 
  ggplot(aes(x = factor(beds), y = factor(bedrooms))) +
  geom_jitter(color = "steelblue") +
  labs(title = "Beds vs Bedrooms",
       x = "Beds",
       y = "Bedrooms")

## Step 3: Data pre-processing ####

# Check for NA's
na_summary = df_train %>% 
  summarise_all(~ sum(is.na(.))) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "na_count") %>% 
  filter(na_count > 0)

# How should we handle NAs?
na_summary

### NA Handling: Host Acceptance Rate ####
# We will impute using mean.
df_train$host_acceptance_rate[is.na(df_train$host_acceptance_rate)] = mean(df_train$host_acceptance_rate, na.rm = TRUE)
df_test$host_acceptance_rate[is.na(df_test$host_acceptance_rate)] = mean(df_train$host_acceptance_rate, na.rm = TRUE)

### NA Handling: Bedrooms ####

# Summarize the number beds in bedrooms
mean_bedrooms = 
  df_train %>% 
  group_by(beds) %>% 
  summarise(mean_bedrooms = mean(bedrooms, na.rm = TRUE))

mean_bedrooms

# We will impute bedrooms using the average 
df_train = 
  df_train %>% 
  left_join(mean_bedrooms, by = "beds") %>% 
  mutate(bedrooms = ifelse(is.na(bedrooms), mean_bedrooms, bedrooms)) %>% 
  select(-mean_bedrooms)

df_test = 
  df_test %>% 
  left_join(mean_bedrooms, by = "beds") %>% 
  mutate(bedrooms = ifelse(is.na(bedrooms), mean_bedrooms, bedrooms)) %>% 
  select(-mean_bedrooms)

### Date handling ####

# Replace host_since with host_days
ref_date = min(df_train$host_since) # Use earliest date in training set as reference point

df_train$host_days = as.numeric(difftime(df_train$host_since, ref_date, units = "days"))
df_test$host_days = as.numeric(difftime(df_test$host_since, ref_date, units = "days"))

# Remove host_since column
df_train = df_train %>% select(! host_since)
df_test = df_test %>% select(! host_since)

## Step 4: Feature Engineering ####

# Normalize data
normalizer = 
  preProcess(
    df_train %>% 
      select(host_acceptance_rate,
             host_total_listings,
             accommodates,
             bathrooms,
             bedrooms,
             beds,
             min_nights,
             total_reviews,
             avg_rating,
             price,
             host_days),
    method = "range"
  )

df_train = predict(normalizer, df_train)
df_test = predict(normalizer, df_test)

## Step 5: Feature & Model Selection ####

### Caret Model ####

nn = train(price ~ .,
           data = df_train,
           method = "nnet",
           trControl = trainControl(method = "cv",
                                    number = 10),
           linout = TRUE,
           trace = FALSE,
           tuneGrid = expand.grid(size = c(1, 2, 3, 5, 10),
                                  decay = c(0.0001,0.001,0.01,0.1)))

nn 

plot(nn)

nn$bestTune

nn$results

varImp(nn)

### Alternative model using neuralnet() ####

# Having issues creating the model with "."
# Create variable with x-variables
df_train_alt = df_train %>% 
  mutate_all(as.numeric)

nn_alt = neuralnet(price ~ superhost + 
                     host_acceptance_rate +
                     host_total_listings +
                     accommodates +
                     bathrooms +
                     bedrooms + 
                     beds +
                     min_nights +
                     total_reviews +
                     avg_rating,
                   data = df_train,
                   linear.output = TRUE,
                   hidden = 2)

nn_alt

plot(nn_alt)


## Step 6: Model Validation ####

### Caret Model ####
# Completed in Step 5 with caret.

### Alternative model using neuralnet() ####

#Shuffle training set
df_shuffle = df_train[sample(nrow(df_train)), ]

# Split data into k folds
folds = cut(seq(1, nrow(df_shuffle)), breaks = 10, labels = FALSE)

# Initialize vectors to store the results
mse_results = c()
rmse_results = c()
r2_results = c()
mae_results = c()

# Cross-validation loop
for (i in 1:10) {
  # Use the fold as the test set
  test_indices = which(folds == i, arr.ind = TRUE)
  test_data = df_shuffle[test_indices, ]
  train_data = df_shuffle[-test_indices, ]

  
  # Use the existing model for prediction
  test_predictions = compute(nn_alt, test_data[, -which(names(test_data) == "price")])$net.result
  
  # Calculate the Mean Squared Error (MSE)
  mse = mean((test_predictions - test_data$price)^2)
  mse_results = c(mse_results, mse)
  
  # Calculate the Root Mean Squared Error (RMSE)
  rmse = sqrt(mse)
  rmse_results = c(rmse_results, rmse)
  
  # Calculate the Mean Absolute Error (MAE)
  mae = mean(abs(test_predictions - test_data$price))
  mae_results = c(mae_results, mae)
  
  # Calculate R-squared
  ss_total = sum((test_data$price - mean(test_data$price))^2)
  ss_residual = sum((test_predictions - test_data$price)^2)
  r2 = 1 - (ss_residual / ss_total)
  r2_results = c(r2_results, r2)
}

# Calculate the average metrics across all folds
average_mse = mean(mse_results)
average_rmse = mean(rmse_results)
average_mae = mean(mae_results)
average_r2 = mean(r2_results)

# Print the results
cat("Average MSE: ", average_mse, "\n")
cat("Average RMSE: ", average_rmse, "\n")
cat("Average MAE: ", average_mae, "\n")
cat("Average R-squared: ", average_r2, "\n")


## Step 7: Predictions and Conclusions ####

predictions = predict(nn, df_test)
postResample(predictions, df_test$price)

predictions_alt = predict(nn_alt, df_test)
postResample(predictions_alt, df_test$price)

# Models had similar performance. Opt for the original model.

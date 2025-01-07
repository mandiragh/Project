# ==============================================================================
# Final Project: Vehicle Prices Regression Model Analysis
# ==============================================================================



# ==============================================================================
# Install Necessary Packages
# ==============================================================================

# List of required packages
required_packages <- c(
  "dplyr",        # Data manipulation
  "ggplot2",      # Visualization
  "carData",      # Diagnostics for multicollinearity (VIF)
  "lmtest",       # Breusch-Pagan test for homoscedasticity
  "fastDummies",  # Dummy encoding
  "Metrics",      # RMSE calculation
  "corrplot",     # Correlation matrix visualization
  "GGally",       # Pairwise plots
  "glmnet",       # Ridge regression
  "caret" ,        # Cross-validation
  "randomForest" 
) 

# Install any missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()
                                    [, "Package"])]
if (length(new_packages)) install.packages(new_packages)

# ==============================================================================
# Load Required Libraries
# ==============================================================================

# Load all necessary libraries
library(dplyr)
library(ggplot2)
library(car)
library(lmtest)
library(fastDummies)
library(Metrics)
library(corrplot)
library(GGally)
library(glmnet)
library(caret)
library(randomForest)

# ==============================================================================
# Load the Vehicle Price Data and Explore Data Structure
# ==============================================================================
# Load the data
VehiclePrices= read.csv(file.choose(), stringsAsFactors = FALSE, sep = ",")
View(VehiclePrices)

# ==============================================================================
#  Explore Data Structure
# ==============================================================================
# Structure of dataset
str(VehiclePrices)

#Summary of the dataset
summary(VehiclePrices)

#Quick descriptive summary using skimr
skimr::skim(VehiclePrices)

#Check for missing values: count missing values per column
colSums(is.na(VehiclePrices))

# Output: No missing values 

sapply(VehiclePrices, function(x) sum(x == "" | x == "NA" | x == "Unknown"))

# ==============================================================================
#  Data Cleaning: 
# ==============================================================================
# Detect Outliers for Price: 
Q1 <- quantile(VehiclePrices$price, 0.25)
Q3 <- quantile(VehiclePrices$price, 0.75)
IQR <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Count and optionally cap outliers
sum(VehiclePrices$price < lower_bound | VehiclePrices$price > upper_bound)  # Count outliers
VehiclePrices$price <- ifelse(VehiclePrices$price < lower_bound, lower_bound, 
                          ifelse(VehiclePrices$price > upper_bound, upper_bound,
                                 VehiclePrices$price))


# Identify duplicate rows in the dataset
duplicates <- VehiclePrices[duplicated(VehiclePrices), ]

# Count the number of duplicate rows
num_duplicates <- nrow(duplicates)

# Display the duplicate rows (if any)
if (num_duplicates > 0) {
  print(paste("Number of duplicate rows:", num_duplicates))
  print("Duplicate rows:")
  print(duplicates)
} else {
  print("No duplicate rows found in the dataset.")
}

# Remove duplicate rows
VehiclePrices <- VehiclePrices[!duplicated(VehiclePrices), ]

# Verify if duplicates are removed
num_duplicates <- sum(duplicated(VehiclePrices))
print(paste("Number of duplicate rows after removal:", num_duplicates))

# ==============================================================================
#  Exploratory Data Analysis(EDA)
# ==============================================================================
summary(VehiclePrices)
sapply(VehiclePrices[, sapply(VehiclePrices, is.character)], function(x) length(unique(x)))

hist(VehiclePrices$price, main = "Histogram of Vehicle Prices", xlab = "Price", col = "skyblue", breaks = 30)
hist(VehiclePrices$km, main = "Histogram of Kilometers", xlab = "Kilometers", col = "lightgreen", breaks = 30)

#Correlation Analysis
numeric_vars <- VehiclePrices[, sapply(VehiclePrices, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
print(cor_matrix)

corrplot(cor_matrix, lab = TRUE)
ggpairs(numeric_vars,
        title = "Pairwise Relationships of Numeric Variables",
        lower = list(continuous = "smooth"),  
        diag = list(continuous = "densityDiag"),  
        upper = list(continuous = "cor"))  

# ==============================================================================
#  Dummy Encoding:
# ==============================================================================
# Load necessary libraries
library(dplyr)
library(fastDummies)

# Create a copy of the dataset to encode
VehiclePrices_encoded <- VehiclePrices

# List of categorical columns
categorical_columns <- names(VehiclePrices_encoded)[sapply(VehiclePrices_encoded, is.character)]

# Print the number of unique values for each categorical column
for (col in categorical_columns) {
  cat(sprintf("%-20s: %d\n", col, length(unique(VehiclePrices_encoded[[col]]))))
}

# Columns to remove (multi-valued categorical columns)
columns_to_remove <- c("Comfort_Convenience", "Entertainment_Media", "Extras", "Safety_Security")

# Remove the above columns from the categorical list
categorical_columns <- setdiff(categorical_columns, columns_to_remove)

# Split and create dummy variables for multi-valued categorical columns
if ("Comfort_Convenience" %in% names(VehiclePrices_encoded)) {
  cc_dummies <- strsplit(as.character(VehiclePrices_encoded$Comfort_Convenience), ",")
  cc_matrix <- do.call(cbind, lapply(unique(unlist(cc_dummies)), function(feature) {
    as.integer(sapply(cc_dummies, function(x) feature %in% x))
  }))
  colnames(cc_matrix) <- paste0("cc_", unique(unlist(cc_dummies)))
  VehiclePrices_encoded <- cbind(VehiclePrices_encoded, as.data.frame(cc_matrix))
}

if ("Entertainment_Media" %in% names(VehiclePrices_encoded)) {
  em_dummies <- strsplit(as.character(VehiclePrices_encoded$Entertainment_Media), ",")
  em_matrix <- do.call(cbind, lapply(unique(unlist(em_dummies)), function(feature) {
    as.integer(sapply(em_dummies, function(x) feature %in% x))
  }))
  colnames(em_matrix) <- paste0("em_", unique(unlist(em_dummies)))
  VehiclePrices_encoded <- cbind(VehiclePrices_encoded, as.data.frame(em_matrix))
}

if ("Extras" %in% names(VehiclePrices_encoded)) {
  ex_dummies <- strsplit(as.character(VehiclePrices_encoded$Extras), ",")
  ex_matrix <- do.call(cbind, lapply(unique(unlist(ex_dummies)), function(feature) {
    as.integer(sapply(ex_dummies, function(x) feature %in% x))
  }))
  colnames(ex_matrix) <- paste0("ex_", unique(unlist(ex_dummies)))
  VehiclePrices_encoded <- cbind(VehiclePrices_encoded, as.data.frame(ex_matrix))
}

if ("Safety_Security" %in% names(VehiclePrices_encoded)) {
  ss_dummies <- strsplit(as.character(VehiclePrices_encoded$Safety_Security), ",")
  ss_matrix <- do.call(cbind, lapply(unique(unlist(ss_dummies)), function(feature) {
    as.integer(sapply(ss_dummies, function(x) feature %in% x))
  }))
  colnames(ss_matrix) <- paste0("ss_", unique(unlist(ss_dummies)))
  VehiclePrices_encoded <- cbind(VehiclePrices_encoded, as.data.frame(ss_matrix))
}

# Drop original multi-valued columns
VehiclePrices_encoded <- VehiclePrices_encoded %>%
  select(-all_of(columns_to_remove))

# Perform one-hot encoding on remaining categorical variables
VehiclePrices_encoded <- fastDummies::dummy_cols(VehiclePrices_encoded, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Convert boolean columns to integers
bool_columns <- sapply(VehiclePrices_encoded, is.logical)
VehiclePrices_encoded[bool_columns] <- lapply(VehiclePrices_encoded[bool_columns], as.integer)

# Preview the structure of the encoded dataset
str(VehiclePrices_encoded)


# ==============================================================================
#  Data Splitting: 
# ==============================================================================
# Set a seed for reproducibility
set.seed(123)

# Define the proportion for the training set (e.g., 70%)
train_index <- sample(1:nrow(VehiclePrices_encoded), 0.7 * nrow(VehiclePrices_encoded))

# Split the data
train_data <- VehiclePrices_encoded[train_index, ]  # 70% for training
test_data <- VehiclePrices_encoded[-train_index, ]  # Remaining 30% for testing

# Check the dimensions of the split datasets
cat("Training Set Size:", nrow(train_data), "rows and", ncol(train_data), "columns\n")
cat("Testing Set Size:", nrow(test_data), "rows and", ncol(test_data), "columns\n")

# Update column names in train_data
colnames(train_data) <- gsub(" ", "_", colnames(train_data))
colnames(train_data) <- gsub("[^[:alnum:]_]", "", colnames(train_data))  # Removes non-alphanumeric characters

# Update column names in test_data
colnames(test_data) <- gsub(" ", "_", colnames(test_data))
colnames(test_data) <- gsub("[^[:alnum:]_]", "", colnames(test_data))

# ==============================================================================
#  Regression Model Development
# ==============================================================================
# Part 1: Step 1: Simple Linear Regression
# ==============================================================================
# Baseline Model: Simple Linear Regression
simple_lm <- lm(price ~ km, data = train_data)

# Summary of the model
summary(simple_lm)

# Predict on test data
test_data$predicted_price_simple <- predict(simple_lm, newdata = test_data)

# Calculate performance metrics
library(Metrics)
rmse_simple <- rmse(test_data$price, test_data$predicted_price_simple)
cat("Simple Linear Regression RMSE:", rmse_simple, "\n")

# R-squared
sse_simple <- sum((test_data$price - test_data$predicted_price_simple)^2)
sst_simple <- sum((test_data$price - mean(test_data$price))^2)
r_squared_simple <- 1 - (sse_simple / sst_simple)
cat("Simple Linear Regression R-squared:", r_squared_simple, "\n")

# ==============================================================================
# Step 2: Diagnostics Checks for Simple Linear Regression
# ==============================================================================

# 1. Residuals vs Fitted Plot
par(mfrow = c(2, 2))  # Layout for multiple plots
plot(simple_lm)

# 2. Histogram of Residuals
hist(residuals(simple_lm), 
     main = "Histogram of Residuals (Simple)", 
     xlab = "Residuals", 
     breaks = 30, 
     col = "lightblue")

# 3. Breusch-Pagan Test for Homoscedasticity
# Null hypothesis: Homoscedasticity (equal variance of residuals)
bptest_simple <- bptest(simple_lm)
cat("Breusch-Pagan Test p-value:", bptest_simple$p.value, "\n")

# 4. Durbin-Watson Test for Autocorrelation
# Null hypothesis: No autocorrelation
durbinWatsonTest_simple <- durbinWatsonTest(simple_lm)
cat("Durbin-Watson Test statistic:", durbinWatsonTest_simple$statistic, "\n")

# ==============================================================================
# Step 3: Influential Points Analysis
# ==============================================================================

# 1. Leverage and Cook's Distance
# Identify high-leverage points
leverage_simple <- hatvalues(simple_lm)
high_leverage_simple <- which(leverage_simple > (2 * mean(leverage_simple)))
cat("High Leverage Points (Simple):", high_leverage_simple, "\n")

# Plot Leverage vs Residuals
plot(leverage_simple, residuals(simple_lm), 
     main = "Leverage vs Residuals (Simple)", 
     xlab = "Leverage", 
     ylab = "Residuals", 
     col = "red", 
     pch = 20)

# Add a horizontal line for leverage threshold
abline(h = 0, col = "blue")

# Cook's Distance
cooksd_simple <- cooks.distance(simple_lm)

# Plot Cook's Distance
plot(cooksd_simple, 
     main = "Cook's Distance (Simple)", 
     xlab = "Observation Index", 
     ylab = "Cook's Distance", 
     col = "skyblue", 
     pch = 20)
abline(h = 4 / nrow(train_data), col = "red", lty = 2)  # Threshold line

# Identify influential points (Cook's Distance > 4/n)
influential_points_simple <- which(cooksd_simple > (4 / nrow(train_data)))
cat("Influential Points (Cook's Distance - Simple):", influential_points_simple, "\n")

# ==============================================================================
# Step 4: Actions to Address Influential Points
# ==============================================================================
# Remove Influential Points if Needed (Optional)
train_data_clean_simple <- train_data[-influential_points_simple, ]

# Refit the Simple Linear Regression model
simple_lm_clean <- lm(price ~ km, data = train_data_clean_simple)

# Summary of the cleaned model
summary(simple_lm_clean)

# Re-calculate performance metrics after removing influential points
test_data$predicted_price_simple_clean <- predict(simple_lm_clean, newdata = test_data)
rmse_clean_simple <- rmse(test_data$price, test_data$predicted_price_simple_clean)
cat("Cleaned Simple Linear Regression RMSE:", rmse_clean_simple, "\n")


# ==============================================================================
# Part 2: Step 1: Multiple Linear Regression
# ==============================================================================
multi_lm <- lm(price ~ km + age + hp_kW + Displacement_cc + cons_comb + 
                 Previous_Owners + Weight_kg, data = train_data)
summary(multi_lm)

#Diagnostic Plot
par(mfrow = c(2, 2))
plot(multi_lm)
# ==============================================================================
# Step 2: Add Interaction and Polynomial Terms
# ==============================================================================
interaction_formula_multi <- price ~ 
  # Main effects
  km + age + hp_kW + Displacement_cc + cons_comb + Previous_Owners + Weight_kg +
  
  # Polynomial terms
  I(km^2) + I(age^2) + I(hp_kW^2) + I(Displacement_cc^2) + I(cons_comb^2) +
  
  # Numeric-Numeric Interactions
  km:age + km:hp_kW + age:hp_kW + hp_kW:Weight_kg + 
  km:Weight_kg + age:Weight_kg + Displacement_cc:Weight_kg +
  cons_comb:hp_kW + cons_comb:km + cons_comb:age +
  
  # Binary Interactions
  km:Inspection_new + age:cc_Air_conditioning + hp_kW:cc_Air_conditioning +
  cons_comb:Inspection_new + Weight_kg:Previous_Owners +
  
  # Count Interactions
  cc_Air_conditioning:cc_Navigation_system + cc_Navigation_system:Inspection_new +
  cc_Air_conditioning:Previous_Owners

multi_lm_interactions <- lm(interaction_formula_multi, data = train_data)
summary(multi_lm_interactions)

# Influential points check
influence_metrics_interactions_multi <- influence.measures(multi_lm_interactions)
summary(influence_metrics_interactions_multi)

# Plot influential points: Cook's Distance
plot(multi_lm_interactions, which = 4)
abline(h = 4 / nrow(train_data), col = "red", lty = 2)

# Leverage (hat values) for interaction model
hat_values_interactions_multi <- hatvalues(multi_lm_interactions)
mean_hat_interactions_multi <- mean(hat_values_interactions_multi)
high_leverage_interactions_multi <- which(hat_values_interactions_multi > 2 * mean_hat_interactions_multi)
cat("High-leverage points (interaction model):", high_leverage_interactions_multi, "\n")

# Plot leverage vs standardized residuals
plot(hat_values_interactions_multi, rstandard(multi_lm_interactions), 
     main = "Leverage vs. Standardized Residuals (Interaction Model)", 
     xlab = "Leverage", ylab = "Standardized Residuals", pch = 20)
abline(h = c(-2, 2), col = "red", lty = 2)
abline(v = 2 * mean_hat_interactions_multi, col = "blue", lty = 2)



# ==============================================================================
# Step 3: Diagnostics Checks
# ==============================================================================
par(mfrow = c(2, 2))
plot(multi_lm_interactions)

# Residuals histogram
hist(residuals(multi_lm_interactions), main = "Histogram of Residuals (Multi)", 
     xlab = "Residuals", breaks = 30)

# Homoscedasticity: Breusch-Pagan test
bptest_multi <- bptest(multi_lm_interactions)

# Independence: Durbin-Watson test
durbinWatsonTest_multi <- durbinWatsonTest(multi_lm_interactions)

# Multicollinearity: VIF
vif_values_multi <- vif(multi_lm_interactions, type ="predictor")
print(vif_values_multi)

# Variables with high VIF
high_vif_multi <- names(vif_values_multi[vif_values_multi > 2.5])
cat("Variables with high VIF (>5):", paste(high_vif_multi, collapse = ", "), "\n")


# ==============================================================================
# Step 4: Model Performance Metrics
# ==============================================================================
test_data$predicted_price_multi <- predict(multi_lm_interactions, newdata = test_data)

# RMSE calculation
rmse_value_multi <- rmse(test_data$price, test_data$predicted_price_multi)
cat("RMSE on test data (Multi):", rmse_value_multi, "\n")

# R-squared calculation
sse_multi <- sum((test_data$price - test_data$predicted_price_multi)^2)
sst_multi <- sum((test_data$price - mean(test_data$price))^2)
r_squared_multi <- 1 - (sse_multi / sst_multi)
cat("R-squared on test data (Multi):", r_squared_multi, "\n")

# ==============================================================================
# Step 5: Remove Influential Points
# ==============================================================================

# Cook's Distance threshold for influential points
cook_threshold <- 4 / nrow(train_data)

# Identify points with high Cook's Distance
high_cooks <- which(cooks.distance(multi_lm_interactions) > cook_threshold)

# Identify points with high leverage
hat_values <- hatvalues(multi_lm_interactions)
mean_hat <- mean(hat_values)
high_leverage <- which(hat_values > 2 * mean_hat)

# Identify points with high standardized residuals
standardized_residuals <- rstandard(multi_lm_interactions)
high_residuals <- which(abs(standardized_residuals) > 2)

# Combine all influential points
influential_points <- unique(c(high_cooks, high_leverage, high_residuals))

cat("Number of influential points identified:", length(influential_points), "\n")
cat("Influential points indices:", influential_points, "\n")

# Remove influential points from the training dataset
train_data_cleaned_multi <- train_data[-influential_points, ]

cat("Number of rows in original data:", nrow(train_data), "\n")
cat("Number of rows in cleaned data:", nrow(train_data_cleaned_multi), "\n")

# ==============================================================================
# Step 6: Stepwise Regression on Cleaned Data
# ==============================================================================

# Fit the empty and full models using the cleaned data
empty_model_cleaned <- lm(price ~ 1, data = train_data_cleaned_multi)
full_model_cleaned <- lm(interaction_formula_multi, data = train_data_cleaned_multi)

# Forward Selection
cat("Performing Forward Selection...\n")
forward_model_cleaned <- step(empty_model_cleaned, 
                              scope = list(lower = empty_model_cleaned, upper = full_model_cleaned), 
                              direction = "forward", trace = 1)
summary(forward_model_cleaned)

# Backward Elimination
cat("Performing Backward Elimination...\n")
backward_model_cleaned <- step(full_model_cleaned, direction = "backward", trace = 1)
summary(backward_model_cleaned)

# Both Directions (Stepwise)
cat("Performing Stepwise Regression (Both Directions)...\n")
stepwise_model_cleaned <- step(empty_model_cleaned, 
                               scope = list(lower = empty_model_cleaned, upper = full_model_cleaned), 
                               direction = "both", trace = 1)
summary(stepwise_model_cleaned)

# ==============================================================================
# Step 7: Refined Clean Model with Diagnostics
# ==============================================================================

# Fit the refined clean model using the best predictors from stepwise regression
clean_model_multi <- lm(formula(forward_model_cleaned), data = train_data_cleaned_multi)
summary(clean_model_multi)


# Influential Points Check for Clean Model
# ==============================================================================

# Cook's Distance threshold for influential points
cook_threshold_clean <- 4 / nrow(train_data_cleaned_multi)

# Plot influential points: Cook's Distance
par(mfrow = c(1, 1))
plot(clean_model_multi, which = 4)  # Cook's distance
abline(h = cook_threshold_clean, col = "red", lty = 2)

# Identify high-leverage points
hat_values_clean_multi <- hatvalues(clean_model_multi)
mean_hat_clean_multi <- mean(hat_values_clean_multi)
high_leverage_clean_multi <- which(hat_values_clean_multi > 2 * mean_hat_clean_multi)
cat("High-leverage points (clean model):", high_leverage_clean_multi, "\n")

# Plot leverage vs standardized residuals
plot(hat_values_clean_multi, rstandard(clean_model_multi), 
     main = "Leverage vs. Standardized Residuals (Clean Model)", 
     xlab = "Leverage", ylab = "Standardized Residuals", pch = 20)
abline(h = c(-2, 2), col = "red", lty = 2)
abline(v = 2 * mean_hat_clean_multi, col = "blue", lty = 2)


# Final Diagnostics Checks for Clean Model
# ==============================================================================

par(mfrow = c(2, 2))
plot(clean_model_multi)

# Breusch-Pagan test for homoscedasticity
bptest_clean_multi <- bptest(clean_model_multi)
cat("Breusch-Pagan Test p-value:", bptest_clean_multi$p.value, "\n")

# Durbin-Watson test for autocorrelation
durbinWatsonTest_clean_multi <- durbinWatsonTest(clean_model_multi)
cat("Durbin-Watson Test statistic:", durbinWatsonTest_clean_multi$statistic, "\n")

# Variance Inflation Factor (VIF) to check multicollinearity
vif_clean_multi <- vif(clean_model_multi, type = "predictor")
print(vif_clean_multi)

# Identify variables with high VIF values
high_vif_clean_multi <- names(vif_clean_multi[vif_clean_multi > 5])
if (length(high_vif_clean_multi) > 0) {
  cat("Variables with high VIF (>5):", paste(high_vif_clean_multi, collapse = ", "), "\n")
} else {
  cat("No variables with high VIF (>5).\n")
}


# Performance Metrics for Clean Model
# ==============================================================================

# Predict on test data using the clean model
test_data$predicted_clean_price_multi <- predict(clean_model_multi, newdata = test_data)

# Calculate RMSE
rmse_clean_multi <- rmse(test_data$price, test_data$predicted_clean_price_multi)
cat("Final RMSE on test data (Clean Model):", rmse_clean_multi, "\n")

# Calculate R-squared
sse_clean_multi <- sum((test_data$price - test_data$predicted_clean_price_multi)^2)
sst_clean_multi <- sum((test_data$price - mean(test_data$price))^2)
r_squared_clean_multi <- 1 - (sse_clean_multi / sst_clean_multi)
cat("Final R-squared on test data (Clean Model):", r_squared_clean_multi, "\n")




# ==============================================================================
# Part 3: Step 1: Polynomial Regression for Nonlinearity
# ==============================================================================
# Polynomial Regression with updated variables
poly_formula_poly <- price ~ km + I(km^2) + age + I(age^2) + hp_kW + I(hp_kW^2) + 
  Displacement_cc + I(Displacement_cc^2) + cons_comb + I(cons_comb^2) + 
  Weight_kg + I(Weight_kg^2) + cc_Air_conditioning + Inspection_new

# Train the model
poly_lm_poly <- lm(poly_formula_poly, data = train_data)

# Summary of the model
summary(poly_lm_poly)

# Predict on test data
test_data$predicted_price_poly <- predict(poly_lm_poly, newdata = test_data)

# Calculate performance metrics
library(Metrics)
rmse_poly <- rmse(test_data$price, test_data$predicted_price_poly)
cat("Polynomial Regression RMSE:", rmse_poly, "\n")

# R-squared
sse_poly <- sum((test_data$price - test_data$predicted_price_poly)^2)
sst_poly <- sum((test_data$price - mean(test_data$price))^2)
r_squared_poly <- 1 - (sse_poly / sst_poly)
cat("Polynomial Regression R-squared:", r_squared_poly, "\n")

# ==============================================================================
# Step 2: Diagnostics Checks for Polynomial Regression
# ==============================================================================
# 1. Residuals vs Fitted Plot
par(mfrow = c(2, 2))  # Layout for multiple plots
plot(poly_lm_poly)

# 2. Histogram of Residuals
hist(residuals(poly_lm_poly), 
     main = "Histogram of Residuals (Poly)", 
     xlab = "Residuals", 
     breaks = 30, 
     col = "lightblue")

# 3. Breusch-Pagan Test for Homoscedasticity
# Null hypothesis: Homoscedasticity (equal variance of residuals)
library(lmtest)
bptest_poly <- bptest(poly_lm_poly)
cat("Breusch-Pagan Test p-value:", bptest_poly$p.value, "\n")

# 4. Durbin-Watson Test for Autocorrelation
# Null hypothesis: No autocorrelation
library(car)
durbinWatsonTest_poly <- durbinWatsonTest(poly_lm_poly)
cat("Durbin-Watson Test statistic:", durbinWatsonTest_poly$statistic, "\n")

# ==============================================================================
# Step 3: Influential Points Analysis
# ==============================================================================
# 1. Leverage and Cook's Distance
# Identify high-leverage points
leverage_poly <- hatvalues(poly_lm_poly)
high_leverage_poly <- which(leverage_poly > (2 * mean(leverage_poly)))
cat("High Leverage Points (Poly):", high_leverage_poly, "\n")

# Plot Leverage vs Residuals
plot(leverage_poly, residuals(poly_lm_poly), 
     main = "Leverage vs Residuals (Poly)", 
     xlab = "Leverage", 
     ylab = "Residuals", 
     col = "red", 
     pch = 20)

# Add a horizontal line for leverage threshold
abline(h = 0, col = "blue")

# Cook's Distance
cooksd_poly <- cooks.distance(poly_lm_poly)

# Plot Cook's Distance
plot(cooksd_poly, 
     main = "Cook's Distance (Poly)", 
     xlab = "Observation Index", 
     ylab = "Cook's Distance", 
     col = "purple", 
     pch = 20)
abline(h = 4 / nrow(train_data), col = "red", lty = 2)  # Threshold line

# Identify influential points (Cook's Distance > 4/n)
influential_points_poly <- which(cooksd_poly > (4 / nrow(train_data)))
cat("Influential Points (Cook's Distance Poly):", influential_points_poly, "\n")

# ==============================================================================
# Part 4: Step 1: Ridge Regression (Regularized Model)
# ==============================================================================
library(glmnet)

# Prepare data for glmnet
x_ridge <- model.matrix(price ~ km + age + hp_kW + Displacement_cc + cons_comb + 
                          Weight_kg + cc_Air_conditioning + Inspection_new, 
                        data = train_data)[, -1]
y_ridge <- train_data$price

# Ridge regression with cross-validation
set.seed(123)  # For reproducibility
ridge_cv <- cv.glmnet(x_ridge, y_ridge, alpha = 0, nfolds = 10)  # alpha = 0 for Ridge

# Optimal lambda
best_lambda_ridge <- ridge_cv$lambda.min
cat("Optimal Lambda for Ridge Regression (Min Error):", best_lambda_ridge, "\n")

# Lambda with 1 Standard Error
lambda_1se_ridge <- ridge_cv$lambda.1se
cat("Lambda with 1SE for Ridge Regression:", lambda_1se_ridge, "\n")

# Visualize Cross-Validation Results
plot(ridge_cv)
title("Ridge Regression Cross-Validation", line = 2.5)

# Train Ridge Model with Optimal Lambda
ridge_final <- glmnet(x_ridge, y_ridge, alpha = 0, lambda = best_lambda_ridge)

# Coefficients at Optimal Lambda
ridge_coeff <- coef(ridge_final)
cat("Ridge Regression Coefficients at Optimal Lambda:\n")
print(ridge_coeff)

# Predict on Test Data
x_test_ridge <- model.matrix(price ~ km + age + hp_kW + Displacement_cc + cons_comb + 
                               Weight_kg + cc_Air_conditioning + Inspection_new, 
                             data = test_data)[, -1]
test_data$predicted_price_ridge <- predict(ridge_final, newx = x_test_ridge)

# Calculate Performance Metrics
library(Metrics)
rmse_ridge <- rmse(test_data$price, test_data$predicted_price_ridge)
cat("Ridge Regression RMSE:", rmse_ridge, "\n")

# R-squared
sse_ridge <- sum((test_data$price - test_data$predicted_price_ridge)^2)
sst_ridge <- sum((test_data$price - mean(test_data$price))^2)
r_squared_ridge <- 1 - (sse_ridge / sst_ridge)
cat("Ridge Regression R-squared:", r_squared_ridge, "\n")

# ==============================================================================
#  Step 2: Additional Steps for Insights
# ==============================================================================
# 1. Analyze Coefficients
ridge_coeff_matrix <- as.matrix(ridge_coeff)
colnames(ridge_coeff_matrix) <- c("Coefficient")
cat("Top Ridge Regression Coefficients:\n")
print(head(ridge_coeff_matrix))

# 2. Compare Performance with Lambda.1SE
ridge_final_1se <- glmnet(x_ridge, y_ridge, alpha = 0, lambda = lambda_1se_ridge)
test_data$predicted_price_ridge_1se <- predict(ridge_final_1se, newx = x_test_ridge)

# RMSE for Lambda.1SE
rmse_ridge_1se <- rmse(test_data$price, test_data$predicted_price_ridge_1se)
cat("Ridge Regression RMSE (Lambda 1SE):", rmse_ridge_1se, "\n")


# ==============================================================================
# Final Model Comparison
# ==============================================================================

# Create a data frame to store the metrics
model_comparison <- data.frame(
  Model = character(),
  RMSE = numeric(),
  R_squared = numeric(),
  stringsAsFactors = FALSE
)

# Add Simple Linear Regression Metrics
model_comparison <- rbind(model_comparison, data.frame(
  Model = "Simple Linear Regression",
  RMSE = rmse_simple,
  R_squared = r_squared_simple
))

# Add Polynomial Regression Metrics
model_comparison <- rbind(model_comparison, data.frame(
  Model = "Polynomial Regression",
  RMSE = rmse_poly,
  R_squared = r_squared_poly
))

# Add Ridge Regression Metrics
model_comparison <- rbind(model_comparison, data.frame(
  Model = "Ridge Regression",
  RMSE = rmse_ridge,
  R_squared = r_squared_ridge
))

# Add Multiple Linear Regression Metrics
model_comparison <- rbind(model_comparison, data.frame(
  Model = "Multiple Linear Regression",
  RMSE = rmse_value_multi,
  R_squared = r_squared_multi
))

# Add Clean Model Metrics
model_comparison <- rbind(model_comparison, data.frame(
  Model = "Clean Model (Stepwise Selection)",
  RMSE = rmse_clean_multi,
  R_squared = r_squared_clean_multi
))

# View the comparison
print(model_comparison)

# ==============================================================================
# Visualize Model Performance
# ==============================================================================

library(ggplot2)

# Plot RMSE comparison
ggplot(model_comparison, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  labs(title = "Model Comparison: RMSE", x = "Model", y = "RMSE") +
  theme_minimal()

# Plot R-squared comparison
ggplot(model_comparison, aes(x = reorder(Model, R_squared), y = R_squared, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  labs(title = "Model Comparison: R-squared", x = "Model", y = "R-squared") +
  theme_minimal()




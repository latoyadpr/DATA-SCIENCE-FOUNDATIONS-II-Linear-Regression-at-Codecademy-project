# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows

# Create a scatter plot of score vs completed

# Show then clear plot

# Fit a linear regression to predict score based on prior lessons completed

# Intercept interpretation:

# Slope interpretation:

# Plot the scatter plot with the line on top

# Show then clear plot

# Predict score for learner who has completed 20 prior lessons

# Calculate fitted values

# Calculate residuals

# Check normality assumption

# Show then clear the plot

# Check homoscedasticity assumption

# Show then clear the plot

# Create a boxplot of score vs lesson

# Show then clear plot

# Fit a linear regression to predict score based on which lesson they took
model = sm.OLS.from_formula('score ~ completed', data=codecademy)
results = model.fit()

# Print the regression coefficients
print(results.params)

# Intercept interpretation:
# The intercept (13.21) is the expected quiz score when the number of completed content items is zero.

# Slope interpretation:
# The slope (1.31) indicates that for each additional content item completed, the quiz score is expected to increase by 1.31 points.

# Create the scatter plot:
plt.scatter(codecademy.completed, codecademy.score)

# Add the regression line:
plt.plot(codecademy.completed, results.predict(codecademy))

# Show then clear the plot
plt.show()
plt.clf()

# Create a new dataset with completed = 20
new_data = pd.DataFrame({'completed': [20]})

# Use the model to predict the score
predicted_score = results.predict(new_data)

# Print the predicted score
print(predicted_score)


# Calculate fitted values
fitted_values = results.predict(codecademy)

# Print the fitted values
print(fitted_values)


# Calculate residuals
residuals = codecademy.score - fitted_values

# Print the residuals
print(residuals)


# Plot a histogram of the residuals
plt.hist(residuals, bins=10, edgecolor='black')

# Show then clear the plot
plt.show()  # Show the plot
plt.clf()   # Clear the plot


# Plot residuals vs fitted values
plt.scatter(fitted_values, residuals)

# Add labels and title
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')

# Show then clear the plot
plt.show()  # Show the plot
plt.clf()   # Clear the plot


# Create a boxplot of score vs lesson
sns.boxplot(x='lesson', y='score', data=codecademy)

# Show then clear the plot
plt.show()  # Show the plot
plt.clf()   # Clear the plot


# Fit a linear regression to predict score based on which lesson they took
model = sm.OLS.from_formula('score ~ lesson', data=codecademy)
results = model.fit()

# Print the regression coefficients
print(results.params)


# Calculate and print the mean quiz score for learners who took Lesson A
mean_score_lesson_a = np.mean(codecademy.score[codecademy.lesson == 'Lesson A'])
print("Mean score for Lesson A:", mean_score_lesson_a)

sns.lmplot(x = 'completed', y = 'score', hue = 'lesson', data = codecademy)
plt.show()


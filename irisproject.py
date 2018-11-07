
# Iris Project
# Data Exploration and Analysis
# By Vishal Kumar Srivastaava

import numpy as np  # Numpy Library is used for numerical operation purpose
import pandas as pd # library to load the dataset
import seaborn as sns # library for Visualization
import matplotlib.pyplot as plt # library for Visualzaiton

# # Iris Data from Seaborn

iris = sns.load_dataset('iris')      # load the dataset from seaborn library and store in dataframe namely:-iris
iris.head()                          # To see top 5 rows of iris
iris.describe()                      # For Statstical Summary of Iris
iris.info()                          # To check datatype ,no of rows,no of columns etc....

############ Visualisation ##################

#pairplot
sns.pairplot(iris, hue='species', size=3, aspect=1);

# Histogram
iris.hist(edgecolor='black', linewidth=1.2, figsize=(12,8));
plt.show();

# Voilin Plot
plt.figure(figsize=(12,8));
plt.subplot(2,2,1)
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species', y='petal_length', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species', y='petal_width', data=iris);

#Boxplot
iris.boxplot(by='species', figsize=(12,8));

# Scatter Matrix
pd.plotting.scatter_matrix(iris, figsize=(12,10))
plt.show()


# # Scikit-Learn API
generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(100)
x.shape
x
y = 1 * x + np.random.randn(100)
y.shape
y
plt.figure(figsize = (10, 8))
plt.scatter(x, y);
# Step 1. Choose a class of model
from sklearn.linear_model import LinearRegression

# Step 2. Choose model hyperparameters
model = LinearRegression(fit_intercept=True)
model

# Step 3. Arrage data into features matrix and target array
X = x.reshape(100, 1)
X.shape

## Step 4. Fit model to data
model.fit(X, y)
model.coef_
model.intercept_
# ## Step 5. Apply trained model to new data
# Creating New Random Data
x_fit = np.linspace(-1, 1)
x_fit
X_fit = x_fit.reshape(-1,1)
x_fit
y_fit = model.predict(X_fit)
y_fit
# ## Visualise
plt.figure(figsize = (10, 8))
plt.scatter(x, y)
plt.plot(x_fit, y_fit);



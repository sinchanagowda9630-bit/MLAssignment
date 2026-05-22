#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Linear Regression using Scikit-Learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot graph
plt.scatter(X, Y, color='blue')
plt.plot(X, Y_pred, color='red')

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression")

plt.show()


# In[ ]:





# In[ ]:





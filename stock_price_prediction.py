#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("jainilcoder/netflix-stock-price-prediction")

print("Path to dataset files:", path)


# In[4]:


import os
import pandas as pd

# See files inside folder
print(os.listdir(path))

# Load CSV (check exact file name from above output)
df = pd.read_csv(path + "/NFLX.csv")

print(df.head())


# In[5]:


df.info()


# In[6]:


data = df.tail(30)


# In[7]:


data['Day'] = np.arange(1, len(data) + 1)


# In[8]:


X = data[['Day']].values
y = data['Close'].values


# In[9]:


degree = 3
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)


# In[10]:


model = LinearRegression()
model.fit(X_poly, y)


# In[11]:


next_day = np.array([[31]])
next_day_poly = poly.transform(next_day)
predicted_price = model.predict(next_day_poly)

print(f"Predicted Netflix closing price for next day: {predicted_price[0]:.2f}")


# In[12]:


X_range = np.linspace(1, 31, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)


# In[13]:


y_pred


# In[14]:


plt.scatter(X, y, label="Actual Price")
plt.plot(X_range, y_pred, label="Polynomial Regression Curve")
plt.scatter(31, predicted_price, label="Predicted Price")
plt.xlabel("Day")
plt.ylabel("Closing Price")
plt.title("Netflix Stock Price Prediction (Polynomial Regression)")
plt.legend()
plt.grid()

plt.show()


# In[15]:


from sklearn.metrics import r2_score

# Predict on training data
y_train_pred = model.predict(X_poly)

# Calculate R² score
r2 = r2_score(y, y_train_pred)

print(f"R² Score: {r2:.4f}")


# In[ ]:





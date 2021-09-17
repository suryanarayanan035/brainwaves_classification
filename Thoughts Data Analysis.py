#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


dataset = pd.read_csv("thoughts/surya left/OpenBCI-RAW-2021-08-18_14-42-49.txt",skiprows=6,header=None)


# In[19]:


dataset=dataset.iloc[:,[1,2,15]]


# In[21]:


dataset=dataset.T


# In[22]:


dataset


# In[32]:


plt.figure(figsize=(10,8))
plt.plot(dataset.iloc[0,:])
plt.show()

# In[33]:



# In[ ]:





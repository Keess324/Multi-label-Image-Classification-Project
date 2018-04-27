
# coding: utf-8

# # Overall EDA Analysis

# In[79]:

# Import Packages
import time
import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pal = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from tqdm import tqdm


# In[39]:

# Test Timer
t = time.time()
# do stuff
elapsed = time.time() - t
elapsed


# ## Loading Label Data

# In[40]:

# Loading Label Data
train_label = pd.read_csv('train_v2.csv')
train_label.head()


# In[41]:

import random
alist = random.sample(range(1, 40479), 25)
alist

train_label.values[34459]


# ## Encoding labels via Dictionary 

# In[42]:

# Count Different Classes "Train y"

tags = train_label['tags'].values
multiclass = [words for segments in tags for words in segments.split()]
class_label = set(multiclass)

print ("labels:")
print(class_label)
print("Numbers of different labels: {} ".format(len(class_label)))


# ## Label Histogram and Distribution Overview

# In[43]:

# label distribution counts /EDA

sumcount = pd.Series(multiclass).value_counts() 
index = sumcount.sort_values(ascending = False).index 
values = sumcount.sort_values(ascending = False).values
print(sumcount)
ax = sns.barplot(y = values, x = index)
plt.xlabel('Labels', fontsize=18)
plt.ylabel('Count', fontsize=18)
ax.set_title("Class Distribution Overview", fontsize = 20)
plt.xticks(rotation=90)
plt.tick_params(labelsize=12)


# ## Image Data Overview

# In[49]:

# Overview the Image and its corresponding labels.
import cv2
import random

styletype = {'grid': False}
plt.rc('axes', **styletype)
_, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(15, 13))

i = 0
alist = random.sample(range(1, 40479), 16)
for m in alist:
    j, k = train_label.values[m]
    name = "AnacondaProjects/train-jpg/"
    extion = ".jpg";
    name += j + extion;
    img = cv2.imread(name)
    ax[i // 4, i % 4].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[i // 4, i % 4].set_title('{} - {}'.format(j, k))
    i += 1    
plt.show()


# In[50]:

# Ordering the label class
# Design a dictionary
def fun(l):
    return [item for sublist in l for item in sublist]

print(class_label)
print("------")
mapping_labels = {i: l for l, i in enumerate(class_label)}  #add counters to iterations
print(mapping_labels)
print("------")
print(mapping_labels.items())
print("------")
order_maplabel = {i: l for l, i in mapping_labels.items()}
print(order_maplabel)


# ## Label Covariance Observation

# In[78]:

# Covariance
labels = train_label['tags'].apply(lambda x: x.split(' '))
com = np.zeros([17]*2) # 17 * 17
for i, j in enumerate(list(sumcount.keys())):
    for m, n in enumerate(list(sumcount.keys())):
        c = 0
        cy = 0
        for row in labels.values:
            if j in row:
                c += 1
                if n in row: cy += 1
        com[i, m] = cy / c

        
data=[go.Heatmap(z=com, x=list(sumcount.keys()), y=list(sumcount.keys()))]
layout=go.Layout(height=800, width=800, title='Covariance Matrix')
fig=dict(data=data, layout=layout)
py.iplot(data, filename='train-com')


# ## Image Dataset Standardization 

# In[52]:

# reshape every image to 64 * 64
# Image Preprocess: Reshape

image_resize = (64, 64)
batch_size = 128

ximage = []
j=0
for i,value in tqdm(train_label.values):
    name = "AnacondaProjects/train-jpg/"
    extion = ".jpg";
    name += i + extion;
    images = cv2.imread(name)
    ximage.append(cv2.resize(images, image_resize))
    j +=1
    

ximage=np.asarray(ximage)    
print(type(ximage))


# ## Label BR

# In[ ]:

# Convert labelled y from string to 0/1 encoding by using above dictionary 
tem = [el[1] for el in train_label.values]

ylabel = []
for i in tqdm(tem):
    tar_y = np.zeros(17)
    str_spl = i.split()
    for t in str_spl:
        tar_y[mapping_labels[t]] = 1
    ylabel.append(tar_y)        
        
    
ylabel=np.asarray(ylabel)  
print(type(ylabel))
print(len(ylabel))


# ## Split Training and Testing

# In[54]:

# Validation and Training Breaking Down 60:40 ===> 24287 : 16192

# Validation and Training Breaking Down 60:40 ===> 24287 : 16192
thre = 0.6
threhold = int(np.floor(thre*len(train_label)))

trainx, validationx, trainy, validationy = ximage[:threhold], ximage[threhold:], ylabel[:threhold], ylabel[threhold:]
              

print(len(trainx))
print(len(trainy))
print(len(validationx))   
print(len(validationy))  
print(len(ylabel))


# ## Overall / Training/ Testing Label Numbers in Each Image Overview

# In[56]:

# ylabel overview

multilabel_train_overview = [sum(i) for i in ylabel]
multilabel_train_overview
print(len(multilabel_train_overview))
print("Training Set:")
print("Number of images with 1 label: {} ".format(multilabel_train_overview.count(1)))
print("Number of images with 2 labels: {} ".format(multilabel_train_overview.count(2)))
print("Number of images with 3 labels: {} ".format(multilabel_train_overview.count(3)))
print("Number of images with 4 labels: {} ".format(multilabel_train_overview.count(4)))
print("Number of images with 5 labels: {} ".format(multilabel_train_overview.count(5)))
print("Number of images with 6 labels: {} ".format(multilabel_train_overview.count(6)))
print("Number of images with 7 labels: {} ".format(multilabel_train_overview.count(7)))
print("Number of images with 8 labels: {} ".format(multilabel_train_overview.count(8)))
print("Number of images with 9 labels: {} ".format(multilabel_train_overview.count(9)))


cnt_train = pd.Series(multilabel_train_overview).value_counts() 
idx_train = cnt_train.sort_values(ascending = False).index 
vls_train = cnt_train.sort_values(ascending = False).values

ax = sns.barplot(y = vls_train, x = idx_train)
plt.xlabel('Labels', fontsize=18)
plt.ylabel('Count', fontsize=18)
ax.set_title("Multi-label overview", fontsize = 20)
plt.xticks(rotation=90)
plt.tick_params(labelsize=12)


# In[16]:

# trainy multi-label counts
trainy
multilabel_train_overview = [sum(i) for i in trainy]
multilabel_train_overview
print(len(multilabel_train_overview))
print("Training Set:")
print("Number of images with 1 label: {} ".format(multilabel_train_overview.count(1)))
print("Number of images with 2 labels: {} ".format(multilabel_train_overview.count(2)))
print("Number of images with 3 labels: {} ".format(multilabel_train_overview.count(3)))
print("Number of images with 4 labels: {} ".format(multilabel_train_overview.count(4)))
print("Number of images with 5 labels: {} ".format(multilabel_train_overview.count(5)))
print("Number of images with 6 labels: {} ".format(multilabel_train_overview.count(6)))
print("Number of images with 7 labels: {} ".format(multilabel_train_overview.count(7)))
print("Number of images with 8 labels: {} ".format(multilabel_train_overview.count(8)))
print("Number of images with 9 labels: {} ".format(multilabel_train_overview.count(9)))


cnt_train = pd.Series(multilabel_train_overview).value_counts() 
idx_train = cnt_train.sort_values(ascending = False).index 
vls_train = cnt_train.sort_values(ascending = False).values

ax = sns.barplot(y = vls_train, x = idx_train)
plt.xlabel('Labels', fontsize=18)
plt.ylabel('Count', fontsize=18)
ax.set_title("Train Multi-label overview", fontsize = 20)
plt.xticks(rotation=90)
plt.tick_params(labelsize=12)


# In[17]:

validationy
multilabel_test_overview = [sum(i) for i in validationy]
multilabel_test_overview
print(len(multilabel_test_overview))

print("Validation Set:")
print("Number of images with 1 label: {} ".format(multilabel_test_overview.count(1)))
print("Number of images with 2 labels: {} ".format(multilabel_test_overview.count(2)))
print("Number of images with 3 labels: {} ".format(multilabel_test_overview.count(3)))
print("Number of images with 4 labels: {} ".format(multilabel_test_overview.count(4)))
print("Number of images with 5 labels: {} ".format(multilabel_test_overview.count(5)))
print("Number of images with 6 labels: {} ".format(multilabel_test_overview.count(6)))
print("Number of images with 7 labels: {} ".format(multilabel_test_overview.count(7)))
print("Number of images with 8 labels: {} ".format(multilabel_test_overview.count(8)))




cnt = pd.Series(multilabel_test_overview).value_counts() 
idx = cnt.sort_values(ascending = False).index 
vls = cnt.sort_values(ascending = False).values

ax = sns.barplot(y = vls, x = idx)
plt.xlabel('Labels', fontsize=18)
plt.ylabel('Count', fontsize=18)
ax.set_title("Validation multi-label Overview", fontsize = 20)
plt.xticks(rotation=90)
plt.tick_params(labelsize=12)


# ## Image Size

# In[18]:

# Input Image Size
trainx.shape


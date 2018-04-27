
# coding: utf-8

# # SIFT Key Point Detection Attempt

# In[238]:

# Loading Package
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


# In[239]:

# Loading Label Data
train_label = pd.read_csv('train_v2.csv')
train_label.head()


# In[242]:

# Opencv Picture Feature

# Img = cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_0.jpg')
# Img = cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_9160.jpg')
Img = cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_20.jpg')

def img_show(img):
    """Convenience function to display a typical color image"""
    plt.grid(False)
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

img_show(Img);


# In[243]:

def gray_img(color_img):
    plt.grid(False)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

Img_gray = gray_img(Img)


plt.imshow(Img_gray, cmap='gray');


# In[245]:

# SIFT Feature Engineering


# Initiate detector
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
kp, des = sift.detectAndCompute(Img_gray, None)


# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(Img_gray,kp,Img,color=(0,255,0), flags=0)
plt.grid(False)
plt.imshow(img2),plt.show()


# # ORB Key Point Detection

# In[ ]:

gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=1000)
find the keypoints with ORB
kp = orb.detect(gray,None)
compute the descriptors with ORB
kp, des = orb.compute(gray, kp)


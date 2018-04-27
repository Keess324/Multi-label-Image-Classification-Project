
# coding: utf-8

# # Second Approach: XGBoost

# In[1]:

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


# In[2]:

# Loading Label Data
train_label = pd.read_csv('train_v2.csv')
train_label.head()


# ## label encoding by using dictionary ( same as CNN approach did)

# In[3]:

# Ordering the label class
# Design a dictionary
tags = train_label['tags'].values
multiclass = [words for segments in tags for words in segments.split()]
class_label = set(multiclass)

def fun(l):
    return [item for sublist in l for item in sublist]

mapping_labels = {i: l for l, i in enumerate(class_label)}  #add counters to iterations
order_maplabel = {i: l for l, i in mapping_labels.items()}


# In[4]:

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


# # Feature Extraction

# ## Sample Image Overview for Feature extraction

# ### RGB channels

# In[7]:

import cv2

# Examples Overview
IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
print(train_label.values[1])
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.grid(False)
plt.colorbar()
plt.show()

print(img.shape)


# import matplotlib.image as mpimg
# img=mpimg.imread('/Users/kechen/AnacondaProjects/train-jpg/train_9160.jpg')
# imgplot = plt.imshow(img)


# In[11]:

IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

plt.imshow(img)
Blue = img
Blue[:,:,0]=0
Blue[:,:,1]=0
plt.imshow(Blue)
plt.grid(False)
plt.show()


# In[12]:

IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

Green = img
Green[:,:,0]=0
Green[:,:,2]=0
plt.imshow(Green)
plt.grid(False)
plt.show()


# In[13]:

IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

Red = img
Red[:,:,1]=0
Red[:,:,2]=0
plt.imshow(Red)
plt.grid(False)
plt.show()


# ### RGB Distributions --- Bimodel

# In[57]:

import statistics
# RGB Feature Extraction
IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
# Red [:,:,0]
# Green [:,:1]
# Blue [:,:,2]


def RGB_features(img):
    # R channal
    R = img[:,:,0].flatten()
    r=img[:,:,0].flatten().mean()
    r_median = statistics.median(R) # Media Of Red Channal
    # Bi-model
    r_leftmodel = R[R < r]
    r_rightmodel = R[R > r]
    # find mode
    r_leftmode = statistics.mode(r_leftmodel)
    r_rightmode = statistics.mode(r_rightmodel)
    
    # Difference 
    r_difmode = abs(r_rightmode - r_leftmode)
    
    print("The mean and median of the red distribution is {} and {}".format(r.round(2), r_median.round(2)))
    print("The bimodel has two modes and difference are {} and {} and {}".format(r_leftmode, r_rightmode, r_difmode))
    plt.hist(R, color='red', bins=100)
    plt.axvline(r, color='brown', linewidth=3)
    plt.axvline(r_median, color="gray", linewidth=3)
    plt.axvline(r_leftmode, color='orange', linewidth=2)
    plt.axvline(r_rightmode, color='orange',linewidth=2)
    plt.show()
    
    
    # G channal
    G = img[:,:,1].flatten()
    g=img[:,:,1].flatten().mean()
    g_median = statistics.median(G) # Media Of Red Channal
    # Bi-model
    g_leftmodel = G[G < g]
    g_rightmodel = G[G > g]
    # find mode 
    g_leftmode = statistics.mode(g_leftmodel)
    g_rightmode = statistics.mode(g_rightmodel)
    
    # Difference 
    g_difmode = abs(g_rightmode - g_leftmode)
    
    print("The mean and median of the Green distribution is {} and {}".format(g.round(2), g_median.round(2)))
    print("The bimodel has two modes and difference are {} and {}  and {}".format(g_leftmode, g_rightmode, g_difmode))
    plt.hist(G, color='green', bins=100)
    plt.axvline(g, color='brown', linewidth=3)
    plt.axvline(g_median, color="gray", linewidth=3)
    plt.axvline(g_leftmode, color='orange', linewidth=2)
    plt.axvline(g_rightmode, color='orange',linewidth=2)
    plt.show()
    
    
    # B channal
    B = img[:,:,2].flatten()
    b=img[:,:,2].flatten().mean()
    b_median = statistics.median(B) # Media Of Red Channal
    # Bi-model
    b_leftmodel = B[B < b]
    b_rightmodel = B[B > b]
    # find mode 
    b_leftmode = statistics.mode(b_leftmodel)
    b_rightmode = statistics.mode(b_rightmodel)
    
    # Difference 
    b_difmode = abs(b_rightmode - b_leftmode)
    
    print("The mean and median of the Blue distribution is {} and {}".format(b.round(2), b_median.round(2) ))
    print("The bimodel has two modes and difference are {} and {} and {}".format(b_leftmode, b_rightmode, b_difmode))
    plt.hist(G, color='blue', bins=100)
    plt.axvline(g, color='brown', linewidth=3)
    plt.axvline(g_median, color="gray", linewidth=3)
    plt.axvline(g_leftmode, color='orange', linewidth=2)
    plt.axvline(g_rightmode, color='orange',linewidth=2)
    plt.show()
    
    return() 

import pandas
import pylab as pl

RGB_features(img)




# ### RGB Feature Extraction Function

# In[142]:

import scipy
import statistics
import numpy
from collections import Counter

def RBG(img):
    R = img[:,:,0].flatten()
    r=img[:,:,0].flatten().mean()
    r_median = statistics.median(R) # Media Of Red Channal
    data = Counter(R)
    r_mode = data.most_common(1)[0][0] # Mode  Of Red Channal
    r_std = numpy.std(R) #Standard Deviation Of Red Channal
    r_max = numpy.max(R) #Maximum Of Red Channal
    r_min = numpy.min(R) #Minmum Of Red Channal
    r_kurtosis = scipy.stats.kurtosis(R) #kurtosis Of Red Channal
    r_skew = scipy.stats.skew(R) #skew Of Red Channal
    
    
    # Bi-model
    r_leftmodel = R[R < r]
    r_rightmodel = R[R > r]
    # find mode
    data = Counter(r_leftmodel)
    r_leftmode = data.most_common(1)[0][0]
    data = Counter(r_rightmodel)
    r_rightmode = data.most_common(1)[0][0] 
    # Difference 
    r_difmode = abs(r_rightmode - r_leftmode)
    
     
    G = img[:,:,1].flatten()
    g=img[:,:,1].flatten().mean() # mean  Of Green Channal
    g_median = statistics.median(G) # Media  Of Green Channal
    data = Counter(G)
    g_mode = data.most_common(1)[0][0] # Mode  Of Green Channal
    g_std = numpy.std(G) #Standard Deviation Of Green Channal
    g_max = numpy.max(G) #Maximum Of Green Channal
    g_min = numpy.min(G) #Minmum Of Green Channal
    g_kurtosis = scipy.stats.kurtosis(G) #kurtosis Of Green Channal
    g_skew = scipy.stats.skew(G) #skew Of Green Channal
    
    
    # Bi-model
    g_leftmodel = G[G < g]
    g_rightmodel = G[G > g]
    # two mode 
    data = Counter(g_leftmodel)
    g_leftmode = data.most_common(1)[0][0]
    
    data = Counter(g_rightmodel)
    g_rightmode = data.most_common(1)[0][0]
    # Difference 
    g_difmode = abs(g_rightmode - g_leftmode)
    
    
    
    B = img[:,:,2].flatten()
    b=img[:,:,2].flatten().mean() # mean of Blue channal
    b_median = statistics.median(B) # Media Of Blue Channal
    data = Counter(B)
    b_mode = data.most_common(1)[0][0] # Mode  Of Blue Channal
    b_std = numpy.std(B) #Standard Deviation Of Blue Channal
    b_max = numpy.max(B) #Maximum Of Blue Channal
    b_min = numpy.min(B) #Minmum  Of Blue Channal
    b_kurtosis = scipy.stats.kurtosis(B) #kurtosis  Of Blue Channal
    b_skew = scipy.stats.skew(B) #skew  Of Blue Channal
    
    # Bi-model
    b_leftmodel = B[B < b]
    b_rightmodel = B[B > b]
    # find mode 
    data = Counter(b_leftmodel)
    b_leftmode = data.most_common(1)[0][0]
    data = Counter(b_rightmodel)
    b_rightmode = data.most_common(1)[0][0]

    # Difference 
    b_difmode = abs(b_rightmode - b_leftmode)
   
    return pd.Series( {"red_mean": r, "red_median": r_median, "red_mode":r_mode, "red_std":r_std, 
                       "red_max":r_max, "red_min": r_min, "red_kurtosis": r_kurtosis, 
                       "red_r_skew": r_skew, "red_leftmode": r_leftmode, 
                       "red_rightmode": r_rightmode, "red_difmode" :r_difmode,
                       "green_mean": g, "green_median": g_median, "green_mode": g_mode, "green_std": g_std, 
                       "green_max": g_max, "green_min": g_min, "green_kurtosis": g_kurtosis,
                       "green_skew": g_skew, "green_leftmode": g_leftmode, "green_rightmode":g_rightmode, 
                       "green_difmode": g_difmode,
                        "blue_mean":b, "blue_median":b_median, "blue_mode":b_mode, "blue_std":b_std, 
                       "blue_max":b_max, "blue_min": b_min, "blue_kurtosis": b_kurtosis,
                       "blue_skew": b_skew, "blue_leftmode": b_leftmode, "blue_rightmode":b_rightmode, 
                       "blue_difmode": b_difmode} )


# ## Sobel Edge Detection

# In[59]:


# Sobel Edge Detection
# loading image
IMG =cv2.imread('/Users/kechen/AnacondaProjects/train-jpg/train_1.jpg')
img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

# converting to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

# Sobel Feature
mag, ang = cv2.cartToPolar(sobelx, sobely)
bins = np.int32(16*ang/(2*np.pi))
bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
hist = np.hstack(hists)







# ### Sobel Feature Extraction Function

# In[114]:

def sob_det (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    # Sobel Feature
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    bins = np.int32(16*ang/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    
    sob=numpy.mean(hist) # mean of Blue channal
    sob_median = statistics.median(hist) # Media Of Blue Channal
    sob_std = numpy.std(hist) #Standard Deviation Of Blue Channal
    sob_max = numpy.max(hist) #Maximum Of Blue Channal
    sob_min = numpy.min(hist) #Minmum  Of Blue Channal
    sob_kurtosis = scipy.stats.kurtosis(hist) #kurtosis  Of Blue Channal
    sob_skew = scipy.stats.skew(hist) #skew  Of Blue Channal
    
    return  pd.Series( {"sob_mean": sob, "sob_median": sob_median, "sob_std":sob_std, 
                        "sob_max": sob_max, "sob_min":sob_min, "sob_kurtosis": sob_kurtosis,
                        "sob_skew":sob_skew })


# ## Overall Feature Extraction( RGB + Sobel)

# In[143]:

import time
t = time.time()
# do stuff


def feature_extranction (img_set):
    Feature = pd.DataFrame([])
    for i in tqdm(img_set):
        names = '/Users/kechen/AnacondaProjects/train-jpg/train_'
        extion = ".jpg";
        names += i + extion;

        IMG =cv2.imread(names)
        img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        
        rbg = RBG(img)
        sob = sob_det (img)
        
        samp = rbg.append(sob)
        Feature =  Feature.append(samp, ignore_index=True)
    return (Feature)
        

        
    

orders = range(0,40479)
img_orders = [str(i) for i in orders]
feature_extranction(img_orders)


elapsed = time.time() - t
elapsed


# # Trasformation from Raw Pixel to Feature Values

# In[159]:

orders = range(0,40479)
img_orders = [str(i) for i in orders]
X = feature_extranction(img_orders)



# In[161]:

X


# In[251]:

Xfinal = X.round(2)
Xfinal.shape
Xfinal[0:1]
Xfinal.head()
# Xfinal.to_csv("X_DF.csv",index=False, encoding='utf-8',header = True)


# ## Split Training and Testing

# In[216]:

# Split Train and validation

thre = 0.6
threhold = int(np.floor(thre*len(train_label)))

trainx, validationx, trainy, validationy = Xfinal[:threhold], Xfinal[threhold:], ylabel[:threhold], ylabel[threhold:]
              
print(trainx.shape)
print(len(trainy))
print(validationx.shape)
print(len(validationy))


# In[213]:

xxx = trainx[:1000]
yyy = trainy[:1000]


# # XGBoost Training model 1

# In[ ]:

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf_multilabel = OneVsRestClassifier(XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100,                               silent=True, objective='binary:logistic', nthread=-1,                               reg_alpha=0, reg_lambda=1,                               base_score=0.5, missing=None))

fit=clf_multilabel.fit(trainx, trainy)


#  OneVsRestClassifier(pred)
# score(xxx, yyy, sample_weight=None)


# ### Score + predictioin model 1

# In[258]:

clf_multilabel.score(trainx,trainy)


# In[259]:

clf_multilabel.score(validationx,validationy)


# In[237]:

XGBy_pred = clf_multilabel.predict_proba(validationx)
type(XGBy_pred)


# # XGBoost Training Model 2

# In[252]:

clf_multilabel2 = OneVsRestClassifier(XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,                               silent=True, objective='binary:logistic', nthread=-1,                               reg_alpha=0, reg_lambda=1,                               base_score=0.5, missing=None))
fit=clf_multilabel2.fit(trainx, trainy)


# ## Score + Precition model 2

# In[253]:

clf_multilabel2.score(trainx,trainy)


# In[257]:

clf_multilabel2.score(validationx,validationy)


# # XGBoost Traiing Model 3

# In[255]:

import time
t = time.time()
clf_multilabel3 = OneVsRestClassifier(XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,                               silent=True, objective='binary:logistic', nthread=-1,                               reg_alpha=0, reg_lambda=1,                               base_score=0.5, missing=None))
fit=clf_multilabel3.fit(trainx, trainy)

elapsed = time.time() - t
elapsed


# ## Score + Prediction  model 3

# In[260]:

clf_multilabel3.score(trainx,trainy)


# In[261]:

clf_multilabel3.score(validationx,validationy)


# # XGBoost Training Model 4

# In[266]:

import time
t = time.time()
clf_multilabel4 = OneVsRestClassifier(XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=100,                               silent=True, objective='binary:logistic', nthread=-1,                               reg_alpha=0, reg_lambda=1,                               base_score=0.5, missing=None))
fit=clf_multilabel4.fit(trainx, trainy)

elapsed = time.time() - t
elapsed


# ## Score + Prediction model 4

# In[267]:

clf_multilabel4.score(trainx,trainy)


# In[268]:

clf_multilabel4.score(validationx,validationy)


# In[269]:

XGBy_pred7 = clf_multilabel4.predict_proba(validationx)


# # Threshold adjustment Section

# In[292]:

print(XGBy_pred[0])
print(validationy[0])
print("-------")
Threshold = 0.65

newtestvali = []
for i in XGBy_pred7:
    i = np.array(i)
    i[i>= Threshold]=1
    i[i<= Threshold]=0
    newtestvali.append(i)
    
# Examples
print(newtestvali[0])
print(validationy[0])


# # Evaluation

# In[293]:

# 1. Exact Match Ratio (MR)

def MR_Measure (target,predict):
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        if i==j:
            msurelist.append(0)
        else:
            msurelist.append(1)


    Exact_Match_Ratio = 1- sum(msurelist)/len(msurelist)

    print("Exact Match Ratio is: {} ".format(Exact_Match_Ratio))
    

MR_Measure(validationy,newtestvali)


# In[294]:

# 2. Hamming Loss (HL)

def HL_measure(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]

    #TP:
    list1 = [x + y for x, y in zip(target_flat, predict_flat)]
    TP = list1.count(2)
    print("True Positives: {} ".format(TP))
    #TN:
    list1 = [x + y for x, y in zip(target_flat, predict_flat)]
    TN = list1.count(0)
    print("True Negatives: {} ".format(TN))
    #FP:
    list2 = [x - y for x, y in zip(target_flat, predict_flat)]
    FP = list2.count(-1)
    print("False Positives: {} ".format(FP))
    #FN:
    list2 = [x - y for x, y in zip(target_flat, predict_flat)]
    FN = list2.count(1)
    print("False Negatives: {} ".format(FN))
    
    #Precision
    Precision = TP/(TP+FP)
    print("Precision is: {} ".format(Precision))
    
    #Recall
    Recall = TP/(TP+FN)
    print("Recall is: {} ".format(Recall))
    
    #Accuracy
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("Accuracy is: {} ".format(Accuracy))  
    
    #F_measure
    F_measure = (TP*2)/(2*TP+FP+FN)
    print("F1_Measure is: {} ".format(F_measure)) 
    
    # Hamming Loss 
    Hamming_Loss = 1-Accuracy

    print("Hamming Loss is: {} ".format(Hamming_Loss))
    

HL_measure(validationy,newtestvali)
    


# In[291]:

# 3. Godbole et Measure (Considering the partical correct)

def GodBole(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]
    print (len(target_flat))
    print (len(predict_flat))
    
    #Accuracy
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = len(list1) - list1.count(0)
        Acc = A_upper/ A_lower
        msurelist.append(Acc)
    GB_Accuracy = sum(msurelist)/len(target)
    print("GodBole Accuracy is: {} ".format(GB_Accuracy)) 
    
    
    #Precision
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = sum(i)
        Pre = A_upper/ A_lower
        msurelist.append(Pre)
    GB_Precision = sum(msurelist)/len(target)
    print("GodBole Precision is: {} ".format(GB_Precision))
    
    #Recall
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        if sum(j) != 0:
            A_lower = sum(j)
            Rec = A_upper/ A_lower
            msurelist.append(Rec)
        else:
            A_lower = 1
            Rec = A_upper/ A_lower
            msurelist.append(Rec)
                
    GB_Recll = sum(msurelist)/len(target)
    print("GodBole Recall is: {} ".format(GB_Recll))
    

    #F_measure
    msurelist=[]
    for i,j in zip(target,predict):
        i = i.astype(int)
        i = i.tolist()
        j = j.astype(int)
        j = j.tolist()
        list1 = [x + y for x, y in zip(i, j)]
        A_upper = list1.count(2)
        A_lower = sum(j)+sum(i)
        F = (2*A_upper)/ A_lower
        msurelist.append(F)
    GB_F_measure = sum(msurelist)/len(target)
    print("GodBole F1_Measure is: {} ".format(GB_F_measure)) 
    
    

GodBole(validationy,newtestvali)
    


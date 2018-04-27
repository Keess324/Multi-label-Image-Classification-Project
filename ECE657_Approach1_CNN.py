
# coding: utf-8

# # First Approach: CNN

# ## Import libraries and Tensorflow backend

# In[1]:

# Import Packages
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

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[3]:

# Loading Label Data
train_label = pd.read_csv('train_v2.csv')
train_label.head()


# # From Here --------------------
# ## Same as EDA part but doing encoding process is mandatory 

# In[4]:

# Count Different Classes "Train y"

tags = train_label['tags'].values
multiclass = [words for segments in tags for words in segments.split()]
class_label = set(multiclass)

print ("labels:")
print(class_label)
print("Numbers of different labels: {} ".format(len(class_label)))


# In[5]:

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


# In[6]:

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


# In[7]:

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


# In[9]:

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


# In[10]:

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


# # Pre-process / Encoding / EDA ends Here-----------------
# 
# ## Training and Testing 60:40

# In[13]:

# Validation and Training Breaking Down 60:40 ===> 24287 : 16192

# Validation and Training Breaking Down 60:40 ===> 24287 : 16192
thre = 0.6
threhold = int(np.floor(thre*len(train_label)))

trainx, validationx, trainy, validationy = ximage[:threhold], ximage[threhold:], ylabel[:threhold], ylabel[threhold:]
              

print(len(trainx))
print(len(validationx))   
print(len(trainy))
print(len(ylabel))
print(len( ylabel[threhold:]))


# In[14]:

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


# In[15]:

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


# In[16]:

# Input Image Size
trainx.shape


# # CNN Modelling 

# ## CNN Model 1

# In[159]:

# Build Models by using CNN
model_1 = Sequential()
model_1.add(Conv2D(64, kernel_size=(3, 3),  # COnv 1
                 activation='relu',
                 input_shape=(64, 64, 3)))

model_1.add(Conv2D(64, (3, 3), activation='relu'))  # Conv 2
model_1.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling 1

model_1.add(Conv2D(128, (3, 3), activation='relu'))  # Conv 3


model_1.add(Dropout(0.5))  # Dropout
model_1.add(Flatten())
model_1.add(Dense(128, activation='relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(17, activation='sigmoid'))

model_1.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])


# In[198]:

model_1.fit(trainx, trainy,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(validationx, validationy))


# ## CNN Model 2 (It is called model 1 in report)

# In[17]:

model_2 = Sequential()
model_2.add(Conv2D(32, kernel_size=(3, 3), strides = 2, # Conv 1
                 activation='relu',
                 input_shape=(64, 64, 3)))

model_2.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling 1

model_2.add(Conv2D(48, (3, 3), activation='relu'))  # Conv 2

model_2.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling 2

model_2.add(Dropout(0.5))  # Dropout
model_2.add(Flatten())
model_2.add(Dense(128, activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(17, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])


# ## CNN Model 2 --- 3 epochs

# In[18]:

model_2.fit(trainx, trainy,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_data=(validationx, validationy))


# ## CNN Model 2 --- 5 epochs

# In[19]:

model_2.fit(trainx, trainy,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(validationx, validationy))


# In[36]:

# Test prediction Results
test_validationy = model_2.predict(validationx)


# ## CNN Model 2--- 10 epochs

# In[20]:

model_2.fit(trainx, trainy,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(validationx, validationy))


# ## CNN model 2----15 epochs 

# In[21]:

model_2.fit(trainx, trainy,
          batch_size=128,
          epochs=15,
          verbose=1,
          validation_data=(validationx, validationy))


# In[22]:

# 15 epochs test results
test_validationy15ep = model_2.predict(validationx)


# ## CNN model 2--- 20 epochs

# In[24]:

model_2.fit(trainx, trainy,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(validationx, validationy))


# In[35]:

# 20 epochs test result
test_validationy20ep = model_2.predict(validationx)


# # Threshold Modification session

# In[66]:

# Evaluations ---- Train  vs. Validation 
print(test_validationy20ep[0])
print(validationy[0])
print("-------")
Threshold = 0.55

newtestvali = []
for i in test_validationy20ep:
    i = np.array(i)
    i[i>= Threshold]=1
    i[i<= Threshold]=0
    newtestvali.append(i)
    
# Examples
print(newtestvali[0])
print(validationy[0])




# ## Evaluations

# In[67]:

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
    


# In[68]:

# 2. Hamming Loss (HL)

def HL_measure(target,predict):
    target_flat = [item for sublist in target for item in sublist]
    predict_flat = [item for sublist in predict for item in sublist]
    print (len(target_flat))
    print (len(predict_flat))

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
    


# In[69]:

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
    


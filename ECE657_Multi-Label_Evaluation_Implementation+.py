
# coding: utf-8

# # Evaluation Implementation 

# ## 1. Exact Match Ratio (MR)

# In[1]:

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


# ## 2. Hamming Loss (HL)

# In[2]:

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


# ## 3. Godbole et Measure (Considering the partical correct)

# In[3]:

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
    
    


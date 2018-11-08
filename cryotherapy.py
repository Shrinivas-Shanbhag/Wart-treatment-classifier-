
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from apyori import apriori  
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf
import pickle
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[2]:


pddtrain=[]
pddtest=[]
a=[]
ytrain=[]
ytest=[]
fuzzy_rules=[]
bell_func_parameters=[]
w1=[]
w2=[]
err1=[]
bias1=[]
bias2=1
o1=[]
i1=[]
x=[]
y=[]
info_of_each_no=[]
association_rules=[]
actualx=[]
actualx1=[]
actualy=[]


# In[48]:


table1=pd.read_csv("1-s2.0-S001048251730001X-mmc6.csv")
#table2=pd.read_csv("1-s2.0-S001048251730001X-mmc4.csv")
# table1.head()


# In[49]:


table1.sample(frac=1)
table1 = table1.sample(frac=1).reset_index(drop=True)
# table1=table1.values
table1=np.asarray(table1)
actualx=table1[:,:-1]
actualy=table1[:,-1:]
# table1


# In[5]:


for i in range(19):
    info_of_each_no.append([])
data=[]
for i in range(table1.shape[0]):
    data.append([])
    for j in range(table1.shape[1]):
        if j==0:
            if table1[i][j]==1:
                data[i].append(2)
                if table1[i][-1]==1:
                    info_of_each_no[2].append(table1[i][j])
            else:
                data[i].append(3)
                if table1[i][-1]==1:
                    info_of_each_no[3].append(table1[i][j])
        elif j==1:
            if table1[i][j]<=21:
                data[i].append(4)
                if table1[i][-1]==1:
                    info_of_each_no[4].append(table1[i][j])
            elif table1[i][j]<=34:
                data[i].append(5)
                if table1[i][-1]==1:
                    info_of_each_no[5].append(table1[i][j])
            else:
                data[i].append(6)
                if table1[i][-1]==1:
                    info_of_each_no[6].append(table1[i][j])
        elif j==2:
            if table1[i][j]<=2:
                data[i].append(7)
                if table1[i][-1]==1:
                    info_of_each_no[7].append(table1[i][j])
            elif table1[i][j]<=4:
                data[i].append(8)
                if table1[i][-1]==1:
                    info_of_each_no[8].append(table1[i][j])
            elif table1[i][j]<=7:
                data[i].append(9)
                if table1[i][-1]==1:
                    info_of_each_no[9].append(table1[i][j])
            else:
                data[i].append(10)
                if table1[i][-1]==1:
                    info_of_each_no[10].append(table1[i][j])
        elif j==3:
            if table1[i][j]<=6:
                data[i].append(11)
                if table1[i][-1]==1:
                    info_of_each_no[11].append(table1[i][j])
            else:
                data[i].append(12)
                if table1[i][-1]==1:
                    info_of_each_no[12].append(table1[i][j])
        elif j==4:
            if table1[i][j]==1:
                data[i].append(13)
                if table1[i][-1]==1:
                    info_of_each_no[13].append(table1[i][j])
            elif table1[i][j]==2:
                data[i].append(14)
                if table1[i][-1]==1:
                    info_of_each_no[14].append(table1[i][j])
            else:
                data[i].append(15)
                if table1[i][-1]==1:
                    info_of_each_no[15].append(table1[i][j])
        elif j==5:
            if table1[i][j]<=110:
                data[i].append(16)
                if table1[i][-1]==1:
                    info_of_each_no[16].append(table1[i][j])
            elif table1[i][j]<=390:
                data[i].append(17)
                if table1[i][-1]==1:
                    info_of_each_no[17].append(table1[i][j])
            else:
                data[i].append(18)
                if table1[i][-1]==1:
                    info_of_each_no[18].append(table1[i][j])
        elif j==6:
            data[i].append(int(table1[i][j]))
            info_of_each_no[int(table1[i][j])].append(table1[i][j])


# In[6]:


for i in range(len(data)):
    x.append(data[i][:-1])
    y.append(data[i][-1])   
x=np.asarray(x)
y=np.asarray(y)
actualx=np.asarray(actualx)
actualy=np.asarray(actualy)
actualx1=actualx


# In[7]:


def gbellmf(x, a, b, c):
    return 1. / (1. + np.abs((x - c) / a) ** (2 * b))


# In[8]:


def infoD(y):
    c1=np.sum(y)
    c2=(y.shape[0]-c1)/y.shape[0]
    c1=c1/y.shape[0]
    return -1*((c1*math.log(c1+1))+(c2*math.log(c2+1)))


# In[9]:


def information_gain(x, y,value_of_attr):
    infod=infoD(y)
    gain=[]
    for i in range(x.shape[1]):
        temp=[[0,0] for i in range(len(value_of_attr[i]))]
        for j in range(x.shape[0]):
            for k in range(len(value_of_attr[i])):
                if x[j][i]==value_of_attr[i][k]:
                    if y[j]==1:
                        temp[k][0]+=1
                    else:
                        temp[k][1]+=1
        result=0
        for j in range(len(value_of_attr[i])):
            t=temp[j][0]+temp[j][1]
            result+=(t/x.shape[0])*(-1*(((temp[j][0]/t)*math.log(temp[j][0]/t+1))+((temp[j][1]/t)*math.log(temp[j][1]/t+1))))
        gain.append(infod-result)      
    return gain    
                


# In[10]:


def feature_selection():
    global pddtrain,pddtest,ytrain,ytest,a,fuzzy_rules,bell_func_parameters,bias1,bias2,w1,w2,i1,o1,err1,actualx,actualy
    xtemp=pddtrain
    ytemp=ytrain
    infogain=information_gain(xtemp,ytemp,[[2,3],[4,5,6],[7,8,9,10],[11,12],[13,14,15],[16,17,18]])
    infogain_list=[]
    for i in range(len(infogain)):
        indx=infogain.index(max(infogain))
        infogain[indx]=0
        infogain_list.append(indx)
    k=5 
    pddtrain=np.delete(pddtrain,infogain_list[k:], axis=1)
    actualx=np.delete(actualx,infogain_list[k:], axis=1)
    a=[[2,3],[4,5,6],[7,8,9,10],[11,12],[13,14,15],[16,17,18]]
    a=np.asarray(a)
    a=np.delete(a,infogain_list[k:], axis=0)
    a=list(a)
    ytrain=ytrain.reshape((ytrain.shape[0],1))
    ytest=ytest.reshape((ytest.shape[0],1))
    data1=np.append(pddtrain,ytrain, axis=1)
    association_rules = apriori(data1, min_support=0.0289, min_confidence=0.6,min_lift=3, min_length=2)
    association_results = list(association_rules)  
    fuzzy_rules=[]
    for i in range(len(association_results)):
        if (0 in list(association_results[i][0])) or (1 in list(association_results[i][0])):
            fuzzy_rules.append(list(association_results[i][0]))
    bell_func_parameters=[]
    for i in range(len(info_of_each_no)):
        if len(info_of_each_no[i])!=0:
            mean=np.mean(info_of_each_no[i])
            bell_func_parameters.append([np.max(info_of_each_no[i])-mean+0.4,4,mean])
        else:
            bell_func_parameters.append([.1,4,-3])
    w1=[[(1.0/(5.0))]*(pddtrain.shape[1]) for i in range(len(fuzzy_rules))]
    w2=[(1.0/7)]*(len(fuzzy_rules))
    i1=[0]*len(fuzzy_rules)
    o1=[0]*len(fuzzy_rules)
    err1=[0]*len(fuzzy_rules)
    bias1=[1]*len(fuzzy_rules)
    bias2=1


# In[11]:


def wxx(wx):
	ww=1/(1+pow( 2.71828,-wx))
	return ww

def wxxx(wx):
	ww=1/(pow( 2.71828,-wx))
	if ww>0.5:
		return 1
	return 0	

def updateabc(x,a,b,c):
    da=(2*b*pow(a,(2*b-1))*pow((x-c),(2*b)))/pow((pow(a,(2*b))+pow((x-c),(2*b))),2)
    dc=(pow(a,(2*b))*2*b*pow((x-c),(2*b-1)))/pow((pow(a,(2*b))+pow((x-c),(2*b))),2)
    try:
        db=(-2*pow(a,(2*b))*pow((x-c),(2*b))*math.log((x-c)/a))/pow((pow(a,(2*b))+pow((x-c),(2*b))),2)
    except:
        db=(-2*pow(a,(2*b))*pow((x-c),(2*b)))/pow((pow(a,(2*b))+pow((x-c),(2*b))),2)
    return [da,db,dc]

def dbydwbar(w,x):
    return (x*pow(2.71828,(w*x)))/pow((pow(2.71828,x)+1),2)
    
    


# In[35]:


def oneiteration():
    global pddtrain,pddtest,ytrain,ytest,a,fuzzy_rules,bell_func_parameters,bias1,bias2,w1,w2,i1,o1,err1
    serror=0
    lrate=0.2
    err1=[0]*len(fuzzy_rules)
    for i in range(pddtrain.shape[0]):
        ##############
        vectr=[0]*19
        for j in range(len(a)):
            for kk in range(len(a[j])):
                vectr[a[j][kk]]=gbellmf(pddtrain[i][j],*bell_func_parameters[a[j][kk]])     
        layer2=[]
        for ii in range(len(fuzzy_rules)):
            val=0.0
            for j in range(len(fuzzy_rules[ii])):
                if fuzzy_rules[ii][j]!=0 and fuzzy_rules[ii][j]!=1:
                    val=(val+vectr[fuzzy_rules[ii][j]])
            layer2.append(val)    
        sum_of_l2=sum(layer2)
        layer3=[]
        for ii in range(len(layer2)):
            layer3.append(layer2[ii]/sum_of_l2)
        ##############
        error1=0
        for ii in range(len(fuzzy_rules)):
            i1[ii]=bias1[ii]
            for j in range(pddtrain.shape[1]):
                i1[ii]+=pddtrain[i][j]*w1[ii][j]
            o1[ii]=wxx(i1[ii]*layer3[ii])
        wx=bias2
        for ii in range(len(fuzzy_rules)):
            wx+=o1[ii]*w2[ii]
        output=wxx(wx)
        error1=ytrain[i]-wxxx(wx)
        error=output*(1-output)*(ytrain[i]-output)
        for ii in range(len(fuzzy_rules)):
            err1[ii]=o1[ii]*(1-o1[ii])*(error*w2[ii])*layer3[ii]
        for ii in range(len(fuzzy_rules)):
            w2[ii]+=(lrate*o1[ii]*error)
        for ii in range(len(fuzzy_rules)):
            for j in range(pddtrain.shape[1]):
                w1[ii][j]+=(lrate*2*layer3[ii]*pddtrain[i][j]*err1[ii])
        ###################################################
        vectr1=[0]*19
        for rule in range(len(fuzzy_rules)):
            val=error*dbydwbar(layer3[rule],i1[rule])*((sum(layer2)-layer2[rule])/pow(sum(layer2),2))
            for j in range(len(fuzzy_rules[rule])):
                if fuzzy_rules[rule][j]!=0 and fuzzy_rules[rule][j]!=1:
                    vectr1[fuzzy_rules[rule][j]]+=val
        learning_rate=.9          
        for j in range(len(a)):
            for kk in range(len(a[j])):
                result=updateabc(pddtrain[i][j],*bell_func_parameters[a[j][kk]])
                bell_func_parameters[a[j][kk]][0]+=result[0]*learning_rate*vectr1[a[j][kk]]
                bell_func_parameters[a[j][kk]][0]+=result[1]*learning_rate*vectr1[a[j][kk]]
                bell_func_parameters[a[j][kk]][0]+=result[2]*learning_rate*vectr1[a[j][kk]]
        ###################################################
        if error1!=0:
            serror+=1
    return serror


# In[36]:


def accuracy():
    global pddtrain,pddtest,ytrain,ytest,a,fuzzy_rules,bell_func_parameters,bias1,bias2,w1,w2,i1,o1,err1
    serror=0
    for i in range(pddtest.shape[0]):
        ################################
        vectr=[0]*19
        for j in range(len(a)):
            for kk in range(len(a[j])):
                vectr[a[j][kk]]=gbellmf(pddtest[i][j],*bell_func_parameters[a[j][kk]])     
        layer2=[]
        for ii in range(len(fuzzy_rules)):
            val=0.0
            for j in range(len(fuzzy_rules[ii])):
                if fuzzy_rules[ii][j]!=0 and fuzzy_rules[ii][j]!=1:
                    val=(val+vectr[fuzzy_rules[ii][j]])
            layer2.append(val)    
        sum_of_l2=sum(layer2)
        layer3=[]
        for ii in range(len(layer2)):
            layer3.append(layer2[ii]/sum_of_l2)
        ######################################
        error1=0
        i1=[0]*len(fuzzy_rules)
        for ii in range(len(fuzzy_rules)):
            i1[ii]=bias1[ii]
            for j in range(pddtest.shape[1]):
                i1[ii]+=pddtest[i][j]*w1[ii][j]
            o1[ii]=wxx(i1[ii]*layer3[ii])
        wx=bias2
        for ii in range(len(fuzzy_rules)):
            wx+=o1[ii]*w2[ii]
        output=wxxx(wx)
        error1=ytest[i]-output
        #################################
        if error1!=0:
            serror+=1
    return serror


# In[42]:


def kfoldcv(iteration_val,nfolds):
    global pddtrain,pddtest,ytrain,ytest,a,fuzzy_rules,bell_func_parameters,bias1,bias2,w1,w2,i1,o1,err1,actualx,actualx1,actualy,x,y
    n=x.shape[0]
    acc=0
    l=0
    avgacc=0.0
    setsize=(n//nfolds)
    err=0
    k=(n//nfolds)
    i=1
    j=k
    while j<n:
        actualx=actualx1
        pddtrain=np.append(x[:i],x[j:],axis=0)
        ytrain=np.append(y[:i],y[j:],axis=0)
        ytest=y[i:j]
        pddtest=x[i:j]
        feature_selection()
        pddtrain=np.append(actualx[:i],actualx[j:],axis=0)
        ytrain=np.append(actualy[:i],actualy[j:],axis=0)
        ytest=actualy[i:j]
        pddtest=actualx[i:j]
        i=j
        j=j+k
        for ii in range(iteration_val):
            err=oneiteration()
            if err==0:
                break;
        racc=accuracy()
        acc1=((pddtest.shape[0]-racc)*1.0)/pddtest.shape[0]
        if acc<acc1:
            acc=acc1
        avgacc+=acc1
        l+=1
        print('accuracy is :',acc1,'best accuracy till now: ',acc)
        print("########################################################################")				
    return [(avgacc*1.0)/l, acc]	

acc=kfoldcv(100,9)		
print("average accuracy is =",acc[0],"best accuracy: ",acc[1])				


# In[43]:


# model = Sequential()
# model.add(Dense(12, input_dim=text_train.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(12, input_dim=text_train.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',)
#               metrics=['accuracy'])

# model.fit(text_train, y_train,
#           epochs=10000,
#           batch_size=128)
# score = model.evaluate(text_test, y_test, batch_size=128)


# In[44]:


# score
# bell_func_parameters


# In[45]:


# for i in range(100):
#     result=oneiteration()
#     print(result)


# In[46]:


# accuracy()


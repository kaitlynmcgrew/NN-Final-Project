#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import csv


# # Section 1: Zillow Housing Data
# 

# In[109]:


zhvi_3brm = pd.read_csv('zillow_data.csv')


# In[110]:


zhvi_3brm.shape


# In[117]:


zhvi_3brm.head()


# In[133]:


for x in zhvi_3brm.columns[9:]:
    i = zhvi_3brm.columns.get_loc(x)
    if i+1>279:
        break
    else:
        zhvi_3brm.iloc[:,9] = zhvi_3brm.iloc[:,9].fillna(zhvi_3brm.iloc[:, i+1])


# In[132]:


zhvi_3brm.isnull().sum()


# In[123]:


zhvi_3brms = zhvi_3brm[['RegionID', 'RegionName', 'State', 'CountyName', '2000-01-31', '2022-08-31']]
zhvi_3brms = zhvi_3brms.rename(columns = {'2000-01-31':'year1', '2022-08-31':'year2'})


# In[124]:


def percent_change(col1, col2):
    return (col2 - col1)/col1


# In[125]:


zhvi_3brms['percent_change'] = zhvi_3brms.apply(lambda x: percent_change(x.year1, x.year2), axis=1)


# In[104]:


zhvi_3brms['city'] = zhvi_3brms['RegionName'].apply(lambda row: row.rstrip('County'))


# In[126]:


zhvi_3brms


# # Section 2: Training & Testing

# In[8]:


# https://www.businessinsider.com/most-miserable-cities-in-the-united-states-based-on-data-2019-9
bi_raw = pd.read_csv('training_cities.csv', header = 1)[['City', 'State', 'WEIGHTED Z-SCORE']].dropna()


# In[9]:


bi_raw.head()


# In[97]:


bi_final = bi_raw.rename(columns = {'City': 'city', 'State':'state', 'WEIGHTED Z-SCORE': 'z'})


# In[19]:


bi_raw


# In[98]:


bi_final['label'] = bi_raw.apply(lambda row: 1 if row.z >= -0.057849734 else 0, axis = 1)
bi_final['city_state'] = bi_final[['city', 'state']].agg(' '.join, axis=1)


# In[99]:


bi_final = bi_final[['city', 'state', 'city_state', 'z' ,'label']]


# In[100]:


bi_final


# # Section 3: News API

# In[52]:


def counter(string, counts):
    x = string.split()
    count = 0
    for i in x:
        if i in counts:
            count += 1
    return count
    


# In[23]:


endpoint="https://newsapi.org/v2/everything"


# In[73]:


positives = ['positive', 'good', 'forward-looking', 
             'productive', 'best', 'finest', 
            'first rate', 'perfect', 'terrific', 'leading', 'optimal', 
             'primo', 'first class', 'greatest']
negatives = ['negative', 'unproductive', 'impotent', 'useless', 
            'disadvantageous', 'unhelpful', 'doubtful', 'fruitless', 'bad', 
             'worst', 'inferior', 'incorrect', 'not right', 'unfavorable']


# In[78]:


results = []


for x in bi_final['city_state'][50:100]:
    URLPost = {'apiKey':'9309e7598f994f3e9f2fc16858a4796c',
            'q':x} #Replace your with your own API key from https://newsapi.org/docs/get-started
    response=requests.get(endpoint, URLPost)
    jsontxt = response.json()
    try:
        if jsontxt['totalResults'] == 0:
            results.append(0)
            continue
        else:
            p = 0
            n = 0
            for i in range(0, 100):
                k = jsontxt['articles'][i]['description']
                p += counter(k, positives)
                n += counter(k, negatives)
            results.append(p/(n+p))
    except:
        results.append(0)
        continue
        
bi_final['score'] = results


# In[102]:


bi_first_50 = bi_final.head(50)


# # Section 4: Simple Neural Network

# In[148]:


def activation_base(x):
    e = 2.718281828459045
    return 1/(1+e**(-x))

activation = np.vectorize(activation_base)

def z_1(x, w1, b1):
    return np.dot(x, w1) + b1


def z_2(z, w2, b2):
    return np.dot(z, w2) + b2

def network(x, w1, b1, w2, b2):
    z1 = z_1(x, w1, b1)
    h = activation(z1)
    z2 = z_2(h, w2, b2)
    yhat = activation(z2)
    return [yhat, z1, z2]

def loss(hat, y):
    return .5*(hat - y)**2
loss_vect = np.vectorize(loss)

def der_sig(x):
    return x*(1-x)
der_sig_vect = np.vectorize(der_sig)

def hat(hats, y):
    return (hats - y)

hat_vect = np.vectorize(hat)

def identity_values(size, x):
    #returns identity matrix of size size with 1's replaced by x values, x should be sizex1 
    return np.identity(size)*np.asarray(x)

def matrix_gb(z1, w2):
    #helper matrix for partial derivatives
    zt = z1.transpose()
    sig = der_sig_vect(activation(zt))
    final = w2.transpose().dot(sig)
    return final.transpose()

def matrix_red(hats, y, Z2):
    #helper matrix for partial derivatives
    yhat = hat_vect(hats, y)
    sigz = identity_values(Z2.shape[0], der_sig_vect(activation(Z2)))
    return identity_values(Z2.shape[0], sigz.dot(yhat))

def dW1(x, w1, gb, r, lr):
    xt = x.transpose()
    final = xt.dot(r.dot(gb))*lr
    return w1-final

def dW2(z1, r, LR, W2):
    h = activation(z1)
    i = np.sum(r.dot(h), axis = 0)
    final = LR*i.transpose()
    return W2-final

def dB(gb, r, LR, B):
    final = np.sum(r.dot(gb), axis = 0)*LR
    return B-final

def dC(r, LR, c):
    final = np.sum(r)*LR
    return c-final


# In[198]:


def full_network(x, w1, b1, w2, b2, lr, y, epoc, report):
    k = 0
    results = network(x, w1, b1, w2, b2)
    l = loss_vect(results[0], y)
    print("result vector of epoc "+str(k)+ " is \n"+str(results[0]))
    print("Mean loss of epoc "+str(k)+ " is \n"+str(np.mean(l)))
    print("Total loss of epoc "+str(k)+ " is \n"+str(np.sum(l)))
    ml = [np.mean(l)]
    sl = [np.sum(l)]
    while k <= epoc:
        gb = matrix_gb(results[1], w2)
        #print("GB matrix at epoc "+str(k)+ " is \n"+ str(gb.shape))
        
        red = matrix_red(results[0], y, results[2])
        #print("red matrix at epoc "+str(k)+ " is \n"+ str(red.shape))
        
        w1 = dW1(x, w1, gb, red, lr)
        if k % report == 0:
            print("w1 matrix at epoc "+str(k)+ " is \n"+ str(w1))
        
        w2 = dW2(results[1], red, lr, w2)
        if k % report == 0:
            print("w2 matrix at epoc "+str(k)+ " is \n"+ str(w2))
        
        b1 = dB(gb, red, lr, b1)
        if k % report == 0:
            print("b1 matrix at epoc "+str(k)+ " is \n"+ str(b1))
        
        b2 = dC(red, lr, b2)
        if k % report == 0:
            print("c matrix at epoc "+str(k)+ " is \n"+ str(b2))
        
        results = network(x, w1, b1, w2, b2)
        
        l = loss_vect(results[0], y)
        if k % report == 0:
            print("result vector of epoc "+str(k)+ " is \n"+str(results[0]))
            print("loss vector of epoc "+str(k)+ " is \n"+str(l))
            print("Mean loss of epoc "+str(k)+ " is \n"+str(np.mean(l)))
            print("Total loss of epoc "+str(k)+ " is \n"+str(np.sum(l)))
        ml.append(np.mean(l))
        sl.append(np.sum(l))
        k += 1
    return [ml, sl, results[0]]


# In[136]:


tester = bi_first_50[['city','city_state', 'z', 'score', 'label']]


# In[230]:


tester.to_csv('tester.csv')


# In[178]:


weight1 = np.random.rand(2, 2)
weight1.shape
weight2 = np.random.rand(2, 1)
bias1 = np.random.rand(1, 2)
bias2 = np.random.rand(1, 1)


# In[179]:


print(x_train.shape)
print(weight1.shape)
print(weight2.shape)
print(bias1.shape)
print(bias2.shape)
print(y_train.shape)


# In[204]:


x = tester[['z', 'score',]]
y = tester['label']


# In[205]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[206]:


y_trainm = np.matrix(y_train.values.reshape(40,1))
x_trainm = np.matrix(x_train.values.reshape(40,2))


# ### Section 4.1: Results

# In[199]:


results = full_network(x_trainm, weight1, bias1, weight2, bias2, .1, y_trainm ,100, 10)


# In[219]:


[x] = results[2].transpose().tolist()


# In[222]:


con = pd.DataFrame({'guess':x, 'actual':y_train})
con['guess_com'] = con.apply(lambda row: 1 if row.guess >= .5 else 0, axis = 1)


# In[225]:


confusion_matrix(con['actual'],con['guess_com'])


# In[231]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')


# In[ ]:





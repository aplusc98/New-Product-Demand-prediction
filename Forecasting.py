
# coding: utf-8
#@author: ASISH CHAKRAPANI
# K-Means Clustering

# Importing the libraries
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Laptopsdata.csv')
data = pd.read_csv('Newdata.csv')
hg = np.array(data)
dataset.loc[576,'Battery_Life':'Sales']= hg
#dataset = pd.concat([dataset, data], axis=0,  ignore_index=True)

if((hg[:,0]<8  and hg[:,0]>0) and (hg[:,1]<2 and hg[:,1]>-1) and (hg[:,2]<1500 and hg[:,2]>0) and (hg[:,3]>-1 and hg[:,3]<2) and (hg[:,4]<3 and hg[:,4]>0.5) and (hg[:,5]>0 and hg[:,5]<16) and (hg[:,6]>10 and hg[:,6]<20) and (hg[:,7]>200 and hg[:,7]<5000)):
    X = dataset.iloc[:,0:8].values
    #find the right number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 20), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    matplotlib.use('Agg')
    fig1 = plt.gcf()
    #plt.show()
    
    fig1.savefig('graph1.png')
    plt.close(fig1)
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters =3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    dataset['Cluster']= y_kmeans
    # finding cluster number of the new data
    l = dataset.iloc[575,9]
    # Grouping by cluster Number
    g=dataset.groupby('Cluster')
    df= g.get_group(l)
    a = df.iloc[:, :-2].values
    b = df.iloc[:, 8].values
    # train test split
    a_train = df.iloc[:-1,:-2].values
    b_train = df.iloc[:-1, 8].values
    a_test = df.iloc[-1:, :-2].values
    b_test = df.iloc[-1: , 8].values
    #model fitting
    #from sklearn.cross_validation import train_test_split
    #a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.1, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(a_train, b_train)
    # Predicting the Test set results
    b_pred = regressor.predict(a_test)
    b_pred = np.round(b_pred)
    print(b_pred)
    y_new_pred=b_pred
    #from sklearn.metrics import mean_squared_error
    #from math import sqrt
    #ms = mean_squared_error(b_test, b_pred)
    #ms1 = sqrt(mean_squared_error(b_test, b_pred ))

    newtech=pd.read_csv('newtech.csv',header=None)
    newtech=np.array(newtech)
    aa1=newtech[:,0]
    aa2=newtech[:,1]
    aa3=newtech[:,2]
    aa4=newtech[:,3]
    aa5=newtech[:,4]
    aa6=newtech[:,5]
    aa7=newtech[:,6]
    aa8=newtech[:,7]
    aa9=newtech[:,8]
    aa10=newtech[:,9]
    aa11=newtech[:,10]
    aa12=newtech[:,11]
    degree=newtech[:,12]
    relad=degree*5*(aa1+aa2+aa3+aa4+aa5)
    
    compat=degree*4*(aa6+aa7+aa8)
    
    complexi=degree*6*(aa9)
    trial=degree*3*aa10
    obser=degree*4*aa11
    maxr=9*5*45
    if(aa7>0):
        maxcp=9*4*21
    else:
        maxcp=9*4*14
    maxc=9*6*7
    maxt=9*3*7
    maxo=9*4*7
    max1=maxr+maxcp+maxc+maxt+maxo
    min1=5+4+4+3+4
    '''normr=(((relad-minr)/(maxr-minr))*(1.25-0.75))+0.75
    normcp=(((compat-mincp)/(maxcp-mincp))*(1.25-0.75))+0.75
    normc=(((complexi-minc)/(maxc-minc))*(1.25-0.75))+0.75
    normt=(((trial-mint)/(maxt-mint))*(1.25-0.75))+0.75
    normo=(((maxo-mino)/(maxo-mino))*(1.25-0.75))+0.75
    '''
    ip=relad+compat+complexi+trial+obser
    ip2=(((ip-min1)/(max1-min1))*(1.25-0.75))+0.75

    #print(min1)
    #print(max1)
    print(ip)
    print(ip2)
    newdemand=int(y_new_pred*ip2)
    import math
    #newdemand=math.ceil(newdemand)
    print(y_new_pred)
    print(newdemand)

    diff=newdemand-y_new_pred
    nd1=newdemand/0.025
    dds=np.array([nd1*0.025,nd1*0.135,nd1*0.34,nd1*0.34,nd1*0.16])
    #dds2=np.array([y_new_pred+(0.025*diff),y_new_pred+(diff*0.135),y_new_pred+(diff*0.34),y_new_pred+(diff*0.34),y_new_pred+(diff*0.16),y_new_pred])
    #dds=np.array([diff*0.025,diff*0.135,diff*0.34,diff*0.34,diff*0.16])

    import pylab as pl
    x=[1,2,3,4,5]
    
    pl.plot(x, dds, "-o", label = "1- early adopters \ 2- jkdbsvjkn")
    for x, y in zip(x, dds):
        pl.text(x, y, str(x), color="black")
    pl.margins(0.1)
    #plt.plot(x,dds,marker='1',linewidth=3,markersize=20,color='blue',markerfacecolor='black',linestyle='dashed',dash_joinstyle='round',dash_capstyle='round')
    #plt.ylabel('SALES')
    plt.xlabel('1-Innovators 2-Early Adopter 3-Early Majority 4-Late Majority 5-Laggards')

    plt.title('Diffusion Curve')
    #plt.legend(loc = 'best')
    

    #plt.plot(range(10))
    #plt.tick_params(
    #axis='x',          # changes apply to the x-axis
    #which='both',      # both major and minor ticks are affected
    #bottom=False,      # ticks along the bottom edge are off
    #top=False,         # ticks along the top edge are off
    #labelbottom=False) # labels along the bottom edge are off
    
    
    plt.ylabel('sales in number of units')
    matplotlib.use('Agg')
    
    fig2 = plt.gcf()
    #plt.show()
    
    fig2.savefig('graph2.png')
    plt.close(fig2)
    #plt.savefig('plot')
    #plt.clf()
    #print(dds2)
    #plt.xlabel()
    #from scipy.interpolate import spline

    #xnew = np.linspace(dds2.min(),dds2.max(),300) #300 represents number of points to make between T.min and T.max

    #power_smooth = spline(dds2,xnew,1)

    #plt.plot(xnew,power_smooth)
    #plt.show()
    #plt.show()

    
    objects = ('Sales based on common features', 'After New features')
    y_pos = np.arange(len(objects))
    performance = [y_new_pred, newdemand]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('sales in number of units')
    plt.title('Demand')
    matplotlib.use('Agg')
    fig3 = plt.gcf()
    fig3.savefig('graph3.png')
    plt.close(fig3)

else:
    print('Invalid parameters')
    


# In[148]:

import pylab


# In[133]:

data = pd.read_csv('Newdata.csv')
hg = np.array(data)
if((hg[:,0]<8  and hg[:,0]>0) and (hg[:,1]<2 and hg[:,1]>-1) and (hg[:,2]<1500 and hg[:,2]>0) and (hg[:,3]>-1 and hg[:,3]<2) and (hg[:,4]<3 and hg[:,4]>0.5) and (hg[:,5]>0 and hg[:,5]<16) and (hg[:,6]>10 and hg[:,6]<20)  and (hg[:,7]>200 and hg[:,7]<1000)):
    print(1)
else:
    print(0)



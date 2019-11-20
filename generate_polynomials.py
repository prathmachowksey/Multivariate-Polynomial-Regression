import numpy as np
import pandas as pd
from itertools import combinations_with_replacement


def get_powers(degree):
    l=[0,1]
    powers=[]
    for i in range(1,degree+1):
        powers.append([x for x in combinations_with_replacement(l,i)])
    powers_flattened=[]
    for sublist in powers:
        for x in sublist:
            powers_flattened.append(x)
    return powers_flattened


def transform_data(X,powers):
    X_new=np.ones((X.shape[0],len(powers)))
    for n in range(X.shape[0]):
        #print(n)
        for i in range(len(powers)):
            for j in powers[i]:
                X_new[n][i]=X_new[n][i]*X[n][j]
    return X_new


#load data
data=pd.read_csv('3D_spatial_network.txt',names=["id","latitude","longitude","altitude"])
print(data.head())

#drop first column
data=data.drop("id",axis=1)
print(data.head())

#normalise data
data=(data-data.mean())/data.std()
print(data.head())

#create X and y matrices

X=data.iloc[:,0:2]
X=X.values

y=data.iloc[:,2:3]
y=y.values  #convert y to numpy array


#get powers
powers_1=get_powers(1)
powers_2=get_powers(2)
powers_3=get_powers(3)
powers_4=get_powers(4)
powers_5=get_powers(5)
powers_6=get_powers(6)



X_1=transform_data(X,powers_1)
X_2=transform_data(X,powers_2)
X_3=transform_data(X,powers_3)
X_4=transform_data(X,powers_4)
X_5=transform_data(X,powers_5)
X_6=transform_data(X,powers_6)

#testing
print(powers_3)
print(X[:3])
print(X_3[:3])

#normalise
y=(y-y.mean())/y.std()
X_1=(X_1-X_1.mean())/X_1.std()
X_2=(X_2-X_2.mean())/X_2.std()
X_3=(X_3-X_3.mean())/X_3.std()
X_4=(X_4-X_4.mean())/X_4.std()
X_5=(X_5-X_5.mean())/X_5.std()
X_6=(X_6-X_6.mean())/X_6.std()
#saving
np.save('X_1.npy',X_1)
np.save('X_2.npy',X_2)
np.save('X_3.npy',X_3)
np.save('X_4.npy',X_4)
np.save('X_5.npy',X_5)
np.save('X_6.npy',X_6)
np.save('y.npy',y)

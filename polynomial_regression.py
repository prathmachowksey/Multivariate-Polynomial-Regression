import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def concatenate_ones(X):
    ones=np.ones((X.shape[0],1))
    X=np.concatenate((ones,X),axis=1) 
    return X

def predicted(X,theta):
    return (X@(np.transpose(theta)))

def cost(X,y,theta):
     return ((1/2)*np.sum(np.square((predicted(X,theta))-y)))

def r_squared(X,y,final_weights):

    return ((np.sum(np.square(predicted(X,final_weights)-y.mean())))/(np.sum(np.square(y-y.mean()))))

def rmse(X,y,final_weights):
    return (np.sqrt(np.square(predicted(X,final_weights)-y).mean()))

def squared_error(X,y,final_weights):
    return np.sum(np.square(predicted(X,final_weights)-y))

def gradient_descent(X,y,theta,learning_rate,iterations):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        
        theta = theta - (learning_rate/len(X)) * np.sum(X * ((predicted(X,theta)) - y), axis=0)
        cost_history[i] = cost(X, y, theta)
        
    return theta,cost_history

def gradient_descent_with_L2_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        
        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((predicted(X,theta)) - y), axis=0)) +  lambda_*theta)
        cost_history[i] = cost(X, y, theta)
        
    return theta,cost_history

def gradient_descent_with_L1_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        temp=np.array(theta)
        
        for j in range(theta.shape[1]):
        
            if(theta[0][j]>=0):
                temp[0][j]=1
            else:
                temp[0][j]=-1
        

        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((predicted(X,theta)) - y), axis=0)) +  lambda_*temp)
        
        cost_history[i] = cost(X, y, theta)
        
        
    return theta,cost_history




def make_model(X,y,learning_rate,iterations,degree):
    print("fitting polynomial of degree: "+str(degree))   
    X=concatenate_ones(X)
    theta=np.ones((1,X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    
    final_weights,cost_history = gradient_descent(X_train,y_train,theta,learning_rate,iterations)
    print("Weights: "+str(final_weights))
    weight_avg=np.absolute(final_weights).mean()
    print("average absolute weight ",weight_avg)

    r2_test=r_squared(X_test,y_test,final_weights)
    print("r2 for testing dataset: "+str(r2_test))
   
    rmse_test=rmse(X_test,y_test,final_weights)
    print("rmse for testing dataset: "+str(rmse_test))

    squared_error_test=squared_error(X_test,y_test,final_weights)
    print("squared_error for testing dataset: "+str(squared_error_test))
    
    temp=np.arange(iterations)
    
    
    plt.plot(temp,cost_history,label='training error')
    
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title(("Training Error for regression model of degree "+str(degree)))
    plt.legend( loc='upper left')

   
    plt.savefig('./degree'+str(degree)+'.png')
    plt.close()
    return r2_test,rmse_test,squared_error_test,weight_avg

def compare_regularisation(X,y,learning_rate,iterations,degree,lambda_):

    
    print("fitting polynomial of degree: "+str(degree))   
    X=concatenate_ones(X)
    theta=np.ones((1,X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print("With L1 regularisation:")
    final_weights_L1_regularisation,cost_history_L1_regularisation = gradient_descent_with_L1_regularisation(X_train,y_train,theta,learning_rate,iterations,lambda_)
    print("Weights: "+str(final_weights_L1_regularisation))
    weight_avg_L1=np.absolute(final_weights_L1_regularisation).mean()
    print("average absolute weight ",weight_avg_L1)

    r2_test_L1_regularisation=r_squared(X_test,y_test,final_weights_L1_regularisation)
    print("r2 for testing dataset: "+str(r2_test_L1_regularisation))
   
    rmse_test_L1_regularisation=rmse(X_test,y_test,final_weights_L1_regularisation)
    print("rmse for testing dataset: "+str(rmse_test_L1_regularisation))

    squared_error_test_L1_regularisation=squared_error(X_test,y_test,final_weights_L1_regularisation)
    print("squared_error for testing dataset: "+str(squared_error_test_L1_regularisation))

    

    print("With L2 regularisation:")

    final_weights_L2_regularisation,cost_history_L2_regularisation = gradient_descent_with_L2_regularisation(X_train,y_train,theta,learning_rate,iterations,lambda_)
    print("Weights: "+str(final_weights_L2_regularisation))
    weight_avg_L2=np.absolute(final_weights_L2_regularisation).mean()
    print("average absolute weight ",weight_avg_L2)


    r2_test_L2_regularisation=r_squared(X_test,y_test,final_weights_L2_regularisation)
    print("r2 for testing dataset: "+str(r2_test_L2_regularisation))
   
    rmse_test_L2_regularisation=rmse(X_test,y_test,final_weights_L2_regularisation)
    print("rmse for testing dataset: "+str(rmse_test_L2_regularisation))

    squared_error_test_L2_regularisation=squared_error(X_test,y_test,final_weights_L2_regularisation)
    print("squared_error for testing dataset: "+str(squared_error_test_L2_regularisation))
    

    return r2_test_L1_regularisation,rmse_test_L1_regularisation,squared_error_test_L1_regularisation,weight_avg_L1,r2_test_L2_regularisation,rmse_test_L2_regularisation,squared_error_test_L2_regularisation,weight_avg_L2


    



X_1=np.load('X_1.npy')
X_2=np.load('X_2.npy')
X_3=np.load('X_3.npy')
X_4=np.load('X_4.npy')
X_5=np.load('X_5.npy')
X_6=np.load('X_6.npy')
y=np.load('y.npy')


r2_list=[]
rmse_list=[]
squared_error_list=[]
weight_avg_list=[]

m=make_model(X_1,y,0.01,1000,1)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])

m=make_model(X_2,y,0.008,1000,2)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])

m=make_model(X_3,y,0.006,1000,3)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])

m=make_model(X_4,y,0.005,1000,4)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])

m=make_model(X_5,y,0.004,1000,5)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])

m=make_model(X_6,y,0.001,1000,6)
r2_list.append(m[0])
rmse_list.append(m[1])
squared_error_list.append(m[2])
weight_avg_list.append(m[3])



plt.plot(np.arange(1,7),r2_list)
plt.xlabel("Degree")
plt.ylabel("R_2")
plt.savefig('./r2'+'.png')
plt.close()

plt.plot(np.arange(1,7),rmse_list)
plt.xlabel("Degree")
plt.ylabel("RMSE")
plt.savefig('./rmse'+'.png')
plt.close()

plt.plot(np.arange(1,7),squared_error_list)
plt.xlabel("Degree")
plt.ylabel("Squared Error")
plt.savefig('./squared_error'+'.png')
plt.close()



#running gradient descent with regularisation for degree 6 polynomial


m=compare_regularisation(X_6,y,0.001,1000,6,1)
r2_list.append(m[0])
r2_list.append(m[4])
rmse_list.append(m[1])
rmse_list.append(m[5])
squared_error_list.append(m[2])
squared_error_list.append(m[6])
weight_avg_list.append(m[3])
weight_avg_list.append(m[4])

temp=['1','2','3','4','5','6','6-L1 reg', '6-L2 reg']

plt.plot(temp,r2_list)
plt.ylabel("R_2")
plt.savefig('./r2_degree_6_regularisation'+'.png')
plt.close()

plt.plot(temp,rmse_list)
plt.ylabel("RMSE")
plt.savefig('./rmse_degree_6_regularisation'+'.png')
plt.close()

plt.plot(temp,squared_error_list)
plt.ylabel("Squared Error")
plt.savefig('./squared_error_degree_6_regularisation'+'.png')
plt.close()

plt.plot(temp,weight_avg_list)
plt.ylabel("Average value of weights")
plt.savefig('./average_weight_degree_6_regularisation'+'.png')
plt.close()

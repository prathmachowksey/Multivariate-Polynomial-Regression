# multivariate-polynomial-regression
Multivariate Polynomial Regression using gradient descent.

In this assignment, polynomial regression models of degrees 1,2,3,4,5,6 have been developed for the [3D Road Network (North Jutland, Denmark) Data Set](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland%2C+Denmark))  using gradient descent method. R squared, RMSE and Squared Error values have been calculated and compared for each model to find the models which best fit the data, as well as ones which overfit the data. L1 and L2 regularisation has been implemented to explore the effect of regularisation on testing loss and overfitting.

## Dataset
- Number of Instances: 43487
- Number of Attributes: 4

### Attributes:
 - OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.
 - LONGITUDE: (Google format) longitude
 - LATITUDE: (Google format) latitude
 - ALTITUDE: Height in meters. 
 
The first attribute(OSM_ID) has been dropped. LONGITUDE and LATITUDE values have been used to predict the target variable, ALTITUDE.

## Structure
The code is divided into two files, generate_polynomials.py and polynomial_regression.py. 
- **generate_polynomials.py** is used to calculate polynomial terms for each degree. For instance, the degree 2 model is of the form: 

<a href="https://www.codecogs.com/eqnedit.php?latex=Y=&space;w0&space;&plus;&space;w1x1&plus;w2x2&plus;w3(x1)^2&space;&plus;w4(x2)^2&plus;w5(x1x2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y=&space;w0&space;&plus;&space;w1x1&plus;w2x2&plus;w3(x1)^2&space;&plus;w4(x2)^2&plus;w5(x1x2)" title="Y= w0 + w1x1+w2x2+w3(x1)^2 +w4(x2)^2+w5(x1x2)" /></a>

The generate_polynomials.py file will calculate the terms <a href="https://www.codecogs.com/eqnedit.php?latex=x1,x2,&space;(x1)^2,(x2)^2,&space;(x1x2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x1,x2,&space;(x1)^2,(x2)^2,&space;(x1x2)" title="x1,x2, (x1)^2,(x2)^2, (x1x2)" /></a>

- **polynomial_regression.py** implements gradient descent for the 6 models which minimises the loss function:

<a href="https://www.codecogs.com/eqnedit.php?latex=E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)^2" title="E= (1/(2*N))\sum_{i=0}^{N} ((w0+ w1x1 + w2x2+...) - Y)^2" /></a>

## Gradient Descent

For each model, the training error was plotted for each iteration. It is clear that the error drops with each iteration. The following figure shows the plot of training error for degree 3 model

![Degree 3](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree3.png)


## R Squared, RMSE and Squared-error
R Squared and RMSE was computed for each model.
![R squared](https://github.com/prathmachowksey/PolynomialRegression/blob/master/r2.png)
![RMSE](https://github.com/prathmachowksey/PolynomialRegression/blob/master/rmse.png)



It follows that up till degree 3, the testing error drops with increasing degree, but increasing degree there after results in an increase in error. This suggests that the degree 3 model best fits the data, where as models of degree 4, 5 and 6 are overfitting the data. The increasing average absolute values of weights with increasing degree also suggests that the weights are assuming arbitrarily large values to fit the data. 

## Regularisation
To address the problem of overfitting, L1 and L2 regularisation has been implemented for the degree 6 model. 
The following figures show the effect of regularisation on testing error.

![RMSE](https://github.com/prathmachowksey/multivariate-polynomial-regression/blob/master/rmse_degree_6_regularisation.png)

Regularisation results in a sharp decrease in testing error. In fact, the loss for degree 6 polynomial model with regularisation is comparable with the loss for  degree 1,2,3 and 4 polynomial models without regularisation.

![Avg Absolute Weight](https://github.com/prathmachowksey/multivariate-polynomial-regression/blob/master/average_weight_degree_6_regularisation.png)

Average absolute weight decreases sharply for the models with regularisation. Once regularised, the ws aren’t assuming large values to cause the model to oscillate wildly and overfit the data.

## Instructions for executing:
Run ```python polynomial_regression.py``` to build models for degrees 1 through 6,generate comparative graphs for R Squared, RMSE and Sqaured Error, using gradient descent with and without regularisation.


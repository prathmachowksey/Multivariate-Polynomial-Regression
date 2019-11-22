# PolynomialRegression
Polynomial Regression using gradient descent.

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

## Regression
The code is divided into two files, generate_polynomials.py and polynomial_regression.py. 
- **generate_polynomials.py** is used to calculate polynomial terms for each degree. For instance, the degree 2 model is of the form: 

<a href="https://www.codecogs.com/eqnedit.php?latex=Y=&space;w0&space;&plus;&space;w1x1&plus;w2x2&plus;w3(x1)^2&space;&plus;w4(x2)^2&plus;w5(x1x2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y=&space;w0&space;&plus;&space;w1x1&plus;w2x2&plus;w3(x1)^2&space;&plus;w4(x2)^2&plus;w5(x1x2)" title="Y= w0 + w1x1+w2x2+w3(x1)^2 +w4(x2)^2+w5(x1x2)" /></a>

The generate_polynomials.py file will calculate the terms <a href="https://www.codecogs.com/eqnedit.php?latex=x1,x2,&space;(x1)^2,(x2)^2,&space;(x1x2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x1,x2,&space;(x1)^2,(x2)^2,&space;(x1x2)" title="x1,x2, (x1)^2,(x2)^2, (x1x2)" /></a>

- **polynomial_regression.py** implements gradient descent for the 6 models which minimises the loss function:
<a href="https://www.codecogs.com/eqnedit.php?latex=E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)2" title="E= (1/(2*N))\sum_{i=0}^{N} ((w0+ w1x1 + w2x2+...) - Y)2" /></a>

### Gradient Descent

For each model, the training error was plotted for each iteration. It is clear that the error drops with each iteration.

![Degree 1](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree1.png)
![Degree 2](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree2.png)
![Degree 3](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree3.png)
![Degree 4](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree4.png)
![Degree 5](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree5.png)
![Degree 6](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree6.png)

R Squared, RMSE and Squared error was computed for each model.
![R squared](https://github.com/prathmachowksey/PolynomialRegression/blob/master/r2.png)
![RMSE](https://github.com/prathmachowksey/PolynomialRegression/blob/master/rmse.png)
![Squared Error](https://github.com/prathmachowksey/PolynomialRegression/blob/master/squared_error.png)




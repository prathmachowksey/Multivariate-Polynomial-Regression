# PolynomialRegression
Polynomial Regression using gradient descent.

In this assignment, polynomial regression models of degrees 1,2,3,4,5,6 have been developed for the [3D Road Network (North Jutland, Denmark) Data Set](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland%2C+Denmark))  using gradient descent method. R squared, RMSE and Squared Error values have been calculated and compared for each model to find the models which best fit the data, as well as ones which overfit the data. L1 and L2 regularisation has been implemented to explore the effect of regularisation on testing loss and overfitting.

## Dataset
-Number of Instances: 43487
-Number of Attributes: 4

### Attributes:
 - OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.
 - LONGITUDE: (Google format) longitude
 - LATITUDE: (Google format) latitude
 - ALTITUDE: Height in meters. 
 
The first attribute(OSM_ID) has been dropped. LONGITUDE and LATITUDE values have been used to predict the target variable, ALTITUDE.

## Regression
The code is divided into two files, generate_polynomials.py and polynomial_regression.py. 
- **generate_polynomials.py** is used to calculate polynomial terms for each degree. For instance, the degree 2 model is of the form: 

Ypred= w0 + w1x1+w2x2+w3(x1)2 +w4(x2)2+w5(x1x2)

The generate_polynomials.py file will calculate the terms x1,x2, (x1)2,(x2)2, (x1x2)

- **polynomial_regression.py** implements gradient descent for the 6 models which minimises the loss function:

E= (1/(2*N))*i=1N((w0+ w1x1 + w2x2+...) - Y)2

For each model, the training error was plotted for each iteration. It is clear that the error drops with each iteration.


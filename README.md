# PolynomialRegression
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
<a href="https://www.codecogs.com/eqnedit.php?latex=E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E=&space;(1/(2*N))\sum_{i=0}^{N}&space;((w0&plus;&space;w1x1&space;&plus;&space;w2x2&plus;...)&space;-&space;Y)2" title="E= (1/(2*N))\sum_{i=0}^{N} ((w0+ w1x1 + w2x2+...) - Y)2" /></a>

## Gradient Descent

For each model, the training error was plotted for each iteration. It is clear that the error drops with each iteration.

![Degree 1](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree1.png)
![Degree 2](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree2.png)
![Degree 3](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree3.png)
![Degree 4](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree4.png)
![Degree 5](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree5.png)
![Degree 6](https://github.com/prathmachowksey/PolynomialRegression/blob/master/degree6.png)

## Weights
- **Degree 1**: [[ 0.0008474   0.14801694 -0.18871601]]
- **Degree 2**: [[ 0.03157033  0.0775256  -0.19877619 -0.19828451  0.05957294  0.02007437]]
- **Degree 3**: [[ 0.08248104  0.51152869  0.41139112 -0.30962312  0.17873276  0.15951618 -0.31064813  0.3150611   0.09436801 -0.47442793]]
- **Degree 4**: [[ 0.47924356  0.58657799  0.41574957  0.52657949  0.61331611  0.79393905 -0.38072313  0.27884217  0.2671658  -0.43045771 -0.30201867  0.10212909 0.07436221 -0.21015514 -0.2768269 ]]
- **Degree 5**: [[ 0.31382851  0.79733443  0.75026668  0.57068641  0.70325388  0.71379482 0.47677718  0.63452489  0.60056857  0.45961866 -0.46276197  0.2602479 0.34386478  0.23206689 -0.07004007 -0.40473914  0.22374914  0.31735775 0.20617249 -0.17940602 -0.67609412]]
- **Degree 6**: [[ 1.43977854  0.81951983  0.79585613  0.86821295  0.8463592   0.89232315 0.58948364  0.73565259  0.74552704  0.59543297  0.65870285  0.72175616 0.72747802  0.63675393  0.61011985 -0.25639376  0.45795429  0.59324287 0.50279404  0.30888324 -0.23218663 -0.36138637  0.19801793  0.40046766 0.35676482  0.16060651 -0.296614   -0.76976689]]

## R Squared, RMSE and Squared-error
R Squared, RMSE and Squared-error was computed for each model.
![R squared](https://github.com/prathmachowksey/PolynomialRegression/blob/master/r2.png)
![RMSE](https://github.com/prathmachowksey/PolynomialRegression/blob/master/rmse.png)
![Squared Error](https://github.com/prathmachowksey/PolynomialRegression/blob/master/squared_error.png)


It follows that up till degree 3, the testing error drops with increasing degree, but increasing degree there after results in an increase in error. This suggests that the degree 3 model best fits the data, where as models of degree 4, 5 and 6 are overfitting the data. The increasing average absolute values of weights with increasing degree also suggests that the weights are assuming arbitrarily large values to fit the data. 

## Regularisation
To address the problem of overfitting, L1 and L2 regularisation has been implemented for the degree 6 model. 



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

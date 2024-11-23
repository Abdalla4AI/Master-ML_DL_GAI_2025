# Master ML DL GAI 2025
Welcome to Our Group for Mastering Machine Learning, Deep Learning, and Generative AI!

## Notes:

### [Chapter1: Exercises & Note 1](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/wiki/Chapter1:-Exercises-&-Note-1)</br>
### [Chapter1: Exercises & Note 2](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/wiki/3.-Chaper1,-Exercises-&-Note-2)


## Chapter 1 – The Machine Learning landscape - modification:
</br>
## Will try to implement "Polynomial Regression model":
</br>
</br>
ref: https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
</br>
</br>

## Fitting the Polynomial Regression model on two components X and y. 

```
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()

lin.fit(X, y)

Fitting the Polynomial Regression model on two components X and y.

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
```

</br>

## Visualising the Linear Regression results using a scatter plot:

```
# Visualising the Linear Regression results
plt.scatter(X, y, color='blue')

#plt.plot(X, lin.predict(X), color='red')
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()
```
</br>

![Alt text of the image](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/images/01.png)

</br>

## Visualize the Polynomial Regression results using a scatter plot:

```
# Visualising the Polynomial Regression results
plt.scatter(X, y, color='blue')

plt.plot(X, lin2.predict(poly.fit_transform(X)),
		color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()
```
</br>

![Alt text of the image](https://github.com/Abdalla4AI/Master-ML_DL_GAI_2025/blob/main/images/02.png)

</br>

## Predict new results with both Linear and Polynomial Regression. Note that the input variable must be in a Numpy 2D array:

```
# Predicting a new result with Linear Regression
# after converting predict variable to 2D array
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)
```
</br>

Results: 
</br>

```
array([[4.8584555]])
```
</br>

```
# Predicting a new result with Polynomial Regression
# after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
```
</br>

Results: 

```
array([[5.63525549]])
```

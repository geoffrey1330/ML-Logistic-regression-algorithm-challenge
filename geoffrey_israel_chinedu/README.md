What is Logistic Regression?
   Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

-->import numpy as np
NumPy is the fundamental package for array computing with Python.

class LogisticRegression:
    
--> def fit(self,X,y,learning_rate=0.001,num_iterations=1000):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
The above code is where the training of the model, logisticRegression happens.
It contains four argument:
X-which is the input feature of the training dataset
y-which is the output feature/label of the training dataset
learning_rate- it metaphorically represents the speed at which a machine learning model "learns".
num_iteration-Iteration is the repetition of a process in order to generate a sequence of outcomes. The sequence will approach                 some end point or end value. Each repetition of the process is a single iteration, and the outcome of each 
              iteration is then the starting point of the next iteration.
n_samples-this is the number of observations,sometimes refer to as the row from the input feature of the training dataset,i.e X.

n_features-this is the number of features,sometimes refer to as the column from the input feature of the training dataset,i.e X.

self.weights-the weight is initialize into zero matrix with a corresponding dimension as that of the features of the training dataset(i.e creating a(1,n)matrix, where n contains 0's ).

self.bias-the bias is also initialized to 0.

Computing the sigmoid activation function:
𝑧=𝑤*X+𝑏--this is a linear_model which is equal to the product of the weight w and input the input feature X with the addition of the bias b.
-->def _sigmoid(self,z):
        return 1/(1+np.exp(-z))
The function above is for computing the sigmoid activation function that returns the value of the sigmoid function of a linear_model,using the formula 1/1+𝑒−(z),where z is the linear_model.


-->for i in range(num_iterations):
            linear_model=np.dot(X,self.weights)+self.bias
            y_predicted=self._sigmoid(linear_model)
            dw=(1/n_samples)+np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)+np.sum(y_predicted-y)
            self.weights -=learning_rate*dw
            self.bias -=learning_rate*db
            
The linear_model variable is gotten from the dot product of weight and the input feature of the training dataset plus the bias.

y_predicted variable is the output of the sigmoid activation function of the linear_model

dw -- gradient of the loss with respect to w, thus same shape as w

db -- gradient of the loss with respect to b, thus same shape as b
where:n_samples is the number of observations or samples.
      X is the input features of our training dataset.
      y_predicted is the output of the sigmoid activation function of the our linear model which is y_predicted.
      y is the output feature/label of our training dataset.
      T is refer to as transpose.    

self.weights -=learning_rate*dw: it updates the weight by multiplying the learning rate with dw.
self.bias -=learning_rate*db:it updates the bias by multiplying the learning rate with db.

This whole processes above is repeated with the number of iteration given.


-->def predict(self,X):
        linear_model=np.dot(X,self.weights)+self.bias
        y_predicted=self._sigmoid(linear_model)
        y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls
The above code is where the testing of the model, logisticRegression happens.
It contains an argument:
X-which is the input feature of the test dataset        
The linear_model variable is gotten from the dot product of weight and the input feature of the test dataset plus the bias.

y_predicted variable is the output of the sigmoid activation function of the linear_model for the test dataset

-->y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
y_predicted_cls should produce value of 1 for all value greater than 0.5 in the y_predicted variable else it should produce 0 for other values.

-->regressor=LogisticRegression()
here the model LogisticRegression have been assigned a new variable regressor

-->from sklearn.model_selection import train_test_split
scikitlearn library for splitting datasets

-->from sklearn import datasets
scikitlearn library that contains inbuilt datasets

-->bc=datasets.load_breast_cancer()
it loads an inbuilt datasets called breast_cancer from the scikitlearn dataset library and assign to a variable bc

-->X,y=bc.data,bc.target
the dataset bc.data is assigned to X and the dataset bc.target is assigned to y

-->x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
the dataset is split into x_train and x_test from the X variable with x_test containing only 20% of the data available

also the dataset is split into y_train and y_test from the y variable with y_test containing only 20% of the data available
 
-->regressor.fit(x_train,y_train,learning_rate=0.05,num_iterations=1000)
the dataset x_train,y_train with a learning_rate of 0.05 and a total of 1000 iteration is used to train the model

-->y_pred=regressor.predict(x_test)
The dataset x_test is used to predict new output y_pred using the model that have been trained

-->from sklearn.metrics import accuracy_score
scikitlearn library used to evaluate a model by comparing two output

-->accuracy_score(y_pred,y_test)
The predicted output y_pred is compared with the original output to measure the performance of the model.
0.9210526315789473
the model produce 92.105% i.e it is approximately 92% accurate with the y_test.






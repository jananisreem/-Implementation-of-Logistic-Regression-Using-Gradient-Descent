<img width="964" height="598" alt="image" src="https://github.com/user-attachments/assets/78f67da4-594f-4da4-a159-3d5a16a1ee6f" /><img width="964" height="598" alt="image" src="https://github.com/user-attachments/assets/bcf19627-a5fd-46a8-b933-0e6ac9f6c6c9" /># Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Generate dataset
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 2. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Initialize parameters
weights = np.zeros(2)
bias = 0
lr = 0.1
epochs = 1000

# 4. Gradient Descent
for _ in range(epochs):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)

    dw = (1/len(X)) * np.dot(X.T, (y_pred - y))
    db = (1/len(X)) * np.sum(y_pred - y)

    weights -= lr * dw
    bias -= lr * db

# 5. Predictions
probabilities = sigmoid(np.dot(X, weights) + bias)
y_pred_final = (probabilities >= 0.5).astype(int)

# 6. Classification Metrics
print("Accuracy:", accuracy_score(y, y_pred_final))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred_final))
print("\nClassification Report:\n", classification_report(y, y_pred_final))

# 7. Plot graph
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

# Decision boundary
x_vals = np.array([min(X[:, 0]), max(X[:, 0])])
y_vals = -(weights[0]*x_vals + bias) / weights[1]

plt.plot(x_vals, y_vals, color='black')

plt.title("Logistic Regression (Gradient Descent)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
## Output:
![logistic regression using gradient descent](sam.png)

<img width="722" height="365" alt="Screenshot 2026-05-25 215203" src="https://github.com/user-attachments/assets/3b1b4208-7994-491e-8f9d-cdb7156360b5" />

<img width="964" height="598" alt="Screenshot 2026-05-25 215211" src="https://github.com/user-attachments/assets/1464e94f-2a20-4214-ac1e-40d148db481e" />

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


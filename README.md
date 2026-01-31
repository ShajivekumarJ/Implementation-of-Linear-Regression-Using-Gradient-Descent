# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SHAJIVE KUMAR J
RegisterNumber:  212225230258
*/
```
~~~
    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("ex3.csv")

x = data['R&D Spend'].values
y = data['Profit'].values

import numpy as np
import matplotlib.pyplot as plt

w = 0.0
b = 0.0
alpha = 0.1
epochs = 100
n = len(x)
x = (x - np.mean(x)) / np.std(x)
losses = []


for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
~~~

## Output:

<img width="846" height="601" alt="Screenshot 2026-01-31 081320" src="https://github.com/user-attachments/assets/3bc292e4-532a-4848-8725-fae408eb59dd" />
<img width="863" height="694" alt="Screenshot 2026-01-31 081340" src="https://github.com/user-attachments/assets/6fc97798-28af-4106-8dc7-849bca478bf7" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

import numpy as np
from wrapper import py_gradient_descent

def main():
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([3, 5, 7, 9, 11], dtype=np.float64)

    learning_rate = 0.01
    iterations = 1000

    result = py_gradient_descent(x, y, learning_rate, iterations)
    print("Result:", result)

if __name__ == "__main__":
 main()

# Gradient Descent in C with Python (Cython Wrapper)

This project demonstrates how to implement **gradient descent in C** for a simple linear regression model and then expose it to Python using **Cython**.  
This way, the heavy lifting is done in **C** for performance, while Python is used for ease of use, testing, and visualization.

---

## 📂 Project Structure

├── GDmod.c # C implementation of gradient descent
├── GDmod.h # C header (struct + function prototypes)
├── wrapper.pyx # Cython wrapper around C functions
├── setup.py # Build script to compile the extension
├── main.py # Python script to run gradient descent


---

## ⚙️ How It Works

### 1. C Code (`GDmod.c` + `GDmod.h`)
Implements **gradient descent** for the equation:


Functions include:
- `descent()` → one gradient descent step  
- `compute_loss()` → mean squared error  
- `gradient_descent()` → runs multiple epochs, tracks `w`, `b`, and loss history  
- `free_losshistory()` → frees allocated memory  

The `GD` struct holds:
```c
typedef struct GD_data {
    float* loss_history;
    float first_weight;
    float last_weight;
    float first_bias;
    float last_bias;
    int epochs;
} GD;


{
  "loss_history": [...],
  "first_weight": ...,
  "last_weight": ...,
  "first_bias": ...,
  "last_bias": ...,
  "epochs": ...
}

3. Build Script (setup.py)

Compiles the C and Cython code into a Python extension:

python setup.py build_ext --inplace
This generates a compiled module (wrapper.*.so on Linux/Mac or wrapper.*.pyd on Windows).


Running the Project
1. Install requirements
pip install cython numpy

2. Compile the extension
python setup.py build_ext --inplace

3. Run the script
python main.py


Key Idea

C → fast numerical computation

Cython → bridge between C and Python

Python → user-friendly interface & visualization

This allows combining performance with ease of use.


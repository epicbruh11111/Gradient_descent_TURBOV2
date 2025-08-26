cdef extern from "GDmod.h":
    ctypedef struct GD:
        float* loss_history
        float first_weight
        float last_weight
        float first_bias
        float last_bias
        int epochs

    void descent(float x[], float y[], int n, float *w, float *b, float learning_rate)
    float compute_loss(float x[], float y[], int n, float w, float b)
    GD gradient_descent(float x[], float y[], int n, float *w, float *b, float learning_rate, int epochs, bint print)
    void free_losshistory(float **loss_history)


import numpy as np
cimport numpy as np

def py_gradient_descent(np.ndarray[np.double_t, ndim=1] x,
                        np.ndarray[np.double_t, ndim=1] y,
                        float w=0, float b=0,
                        float learning_rate=0.01,
                        int epochs=100,
                        bint print=False):

    cdef float cw = w
    cdef float cb = b
    cdef int n = x.shape[0]

    cdef GD result = gradient_descent(<float*>x.data,
                                     <float*>y.data,
                                     n,
                                     &cw,
                                     &cb,
                                     learning_rate,
                                     epochs,
                                     print)
    
    # convert C array to Python list
    py_loss = [result.loss_history[i] for i in range(result.epochs)]
    
    # free memory
    free_losshistory(&result.loss_history)
    
    return {
        "loss_history": py_loss,
        "first_weight": result.first_weight,
        "last_weight": result.last_weight,
        "first_bias": result.first_bias,
        "last_bias": result.last_bias,
        "epochs": result.epochs
    }

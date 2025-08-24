#include <stdio.h>
#include <stdlib.h>
#include "GDmod.h"

void descent(float x[], float y[], int n, float *w, float *b, float learning_rate) {
    float dldw = 0.0;
    float dldb = 0.0;
    for (int i = 0; i < n; i++) {
        float error = y[i] - (*w * x[i] + *b);
        dldw += -2 * x[i] * error;
        dldb += -2 * error;
    }
    *w -= learning_rate * (dldw / n);
    *b -= learning_rate * (dldb / n);
}

float compute_loss(float x[], float y[], int n, float w, float b) {
    float loss = 0.0;
    for (int i = 0; i < n; i++) {
        float error = y[i] - (w * x[i] + b);
        loss += error * error;
    }
    return loss / n;
}

GD gradient_descent(float x[], float y[], int n, float *w, float *b, float learning_rate, int epochs, bool print) {
    GD Data;
    Data.epochs = epochs;
    Data.loss_history = malloc(sizeof(float) * epochs);

    if (Data.loss_history == NULL || w == NULL || b == NULL) {
        fprintf(stderr, "ERROR: malloc failed or null pointer passed.\n");
        GD empty = {0};
        empty.loss_history = NULL;
        return empty;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        descent(x, y, n, w, b, learning_rate);
        float loss = compute_loss(x, y, n, *w, *b);
        Data.loss_history[epoch] = loss;

        if (epoch == 0) {
            Data.first_weight = *w;
            Data.first_bias = *b;
        }

        if (epoch == epochs - 1) {
            Data.last_weight = *w;
            Data.last_bias = *b;
        }

        if (print) {
            printf("Epoch %d | loss: %f | w: %f | b: %f\n", epoch, loss, *w, *b);
        }
    }

    return Data;
}

void free_losshistory(float **loss_history) {
    if (loss_history && *loss_history) {
        free(*loss_history);
        *loss_history = NULL;
    }
}

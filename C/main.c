#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Structure to hold gradient descent data
typedef struct GD_data {
    float* loss_history;    // Stores loss for each epoch
    float first_weight;     // Weight after first update
    float last_weight;      // Weight after last update
    float first_bias;       // Bias after first update
    float last_bias;        // Bias after last update
    int epochs;             // Number of epochs
} GD;

// Performs a single step of gradient descent
void descent(float x[], float y[], int n, float *w, float *b, float learning_rate) {
    float dldw = 0.0;
    float dldb = 0.0;

    for (int i = 0; i < n; i++) {
        float error = y[i] - (*w * x[i] + *b);
        dldw += -2 * x[i] * error;
        dldb += -2 * error;
    }

    // Update weight and bias
    *w -= learning_rate * (dldw / n);
    *b -= learning_rate * (dldb / n);
}

// Computes mean squared error for current weight and bias
float compute_loss(float x[], float y[], int n, float w, float b) {
    float loss = 0.0;
    for (int i = 0; i < n; i++) {
        float error = y[i] - (w * x[i] + b);
        loss += error * error;
    }
    return loss / n;
}

// Performs gradient descent for multiple epochs
GD gradient_descent(float x[], float y[], int n, float *w, float *b, float learning_rate, int epochs, bool print) {
    GD Data;
    Data.epochs = epochs;
    Data.loss_history = malloc(sizeof(float) * epochs);

    // Check malloc success and input pointers
    if (Data.loss_history == NULL || w == NULL || b == NULL) {
        fprintf(stderr, "ERROR: malloc failed or null pointer passed.\n");
        // Return empty struct with NULL pointer to indicate failure
        GD empty = {0};
        empty.loss_history = NULL;
        return empty;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        descent(x, y, n, w, b, learning_rate);
        float loss = compute_loss(x, y, n, *w, *b);

        // Store loss at current epoch (indexing is 0-based)
        Data.loss_history[epoch] = loss;

        // Save first weight/bias after first update
        if (epoch == 0) {
            Data.first_weight = *w;
            Data.first_bias = *b;
        }

        // Save last weight/bias after final update
        if (epoch == epochs - 1) {
            Data.last_weight = *w;
            Data.last_bias = *b;
        }

        // Optional: print progress
        if (print) {
            printf("Epoch %d | loss: %f | w: %f | b: %f\n", epoch, loss, *w, *b);
        }
    }

    return Data;
}

// Frees loss history and sets caller pointer to NULL to prevent dangling pointer
void free_losshistory(float **loss_history) {
    if (loss_history && *loss_history) {
        free(*loss_history);
        *loss_history = NULL;
    }
}

int main() {
    // Example dataset
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {2.0, 4.0, 6.0};
    int n = sizeof(x) / sizeof(x[0]);

    float w = 0.0;               // Initial weight
    float b = 0.0;               // Initial bias
    float learning_rate = 0.01;  // Step size for gradient descent
    int epochs = 400;

    // Run gradient descent
    GD d = gradient_descent(x, y, n, &w, &b, learning_rate, epochs, false);

    // Check if malloc succeeded before accessing loss_history
    if (d.loss_history != NULL) {
        for (int i = 0; i < d.epochs; i++) {
            printf("Epoch %d loss: %f\n", i, d.loss_history[i]);
        }
        // Free allocated memory to prevent leaks
        free_losshistory(&d.loss_history);
    } else {
        fprintf(stderr, "Gradient descent failed.\n");
    }

    return 0;
}

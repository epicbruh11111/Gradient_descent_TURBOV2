#ifndef GDMOD_H
#define GDMOD_H

#include <stdbool.h>

// Structure to hold gradient descent data
typedef struct GD_data {
    float* loss_history;    // Stores loss for each epoch
    float first_weight;     // Weight after first update
    float last_weight;      // Weight after last update
    float first_bias;       // Bias after first update
    float last_bias;        // Bias after final update
    int epochs;             // Number of epochs
} GD;

// Function prototypes
void descent(float x[], float y[], int n, float *w, float *b, float learning_rate);
float compute_loss(float x[], float y[], int n, float w, float b);
GD gradient_descent(float x[], float y[], int n, float *w, float *b, float learning_rate, int epochs, bool print);
void free_losshistory(float **loss_history);

#endif

#include <stdbool.h>
#include <stdio.h>

// Gradient descent single step
void descent(float x[], float y[], int n, float *w, float *b, float learning_rate) {
    float dldw = 0.0;
    float dldb = 0.0;

    for (int i = 0; i < n; i++) {
        float error = y[i] - (*w * x[i] + *b);
        dldw += -2 * x[i] * error;
        dldb += -2 * error;
    }

    *w = *w - learning_rate * (dldw / n);
    *b = *b - learning_rate * (dldb / n);
}

// Calculate mean squared error
float compute_loss(float x[], float y[], int n, float w, float b) {
    float loss = 0.0;
    for (int i = 0; i < n; i++) {
        float error = y[i] - (w * x[i] + b);
        loss += error * error;
    }
    return loss / n;
}

// all the new allocated data will be stored here after finishing the calculation for the user to use
typedef struct GD_data{
    float w;
    float b;
    float loss;
    int   epoch;
    int   lenght_x;

} GD;

// Iteratively perform gradient descent
void gradient_descent(float x[], float y[], int n, float *w, float *b, float learning_rate, int epochs,bool print) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        descent(x, y, n, w, b, learning_rate);
        float loss = compute_loss(x, y, n, *w, *b);
        if (print == true){
            printf("%d loss: %f, w: %f, b: %f\n", epoch, loss, *w, *b);
        }
    }

}

int main() {
    // Example data
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {2.0, 4.0, 6.0};
    int n = sizeof(x)/sizeof(x[0]);
    float w = 0.0;
    float b = 0.0;
    float learning_rate = 0.01;
    int epochs = 400;

    gradient_descent(x, y, n, &w, &b, learning_rate, epochs, true);

    return 0;
}

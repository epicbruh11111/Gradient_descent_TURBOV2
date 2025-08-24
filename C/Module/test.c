#include "GDmod.h"



int main(){

    // Example dataset
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {2.0, 4.0, 6.0};
    int n = sizeof(x) / sizeof(x[0]);

    float w = 0.0;               // Initial weight
    float b = 0.0;               // Initial bias
    float learning_rate = 0.01;  // Step size for gradient descent
    int epochs = 400;


    GD d = gradient_descent(x, y, n, &w, &b, learning_rate, epochs, true);

    return 0;
}
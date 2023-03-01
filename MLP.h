#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

using namespace std;

class Perceptron
{
  public:
    vector<double> weights;
    double bias;
    Perceptron(int inputs, double bias=1.0);
    double run(vector<double> x);
    void set_weights(vector<double> w_init);
    double sigmoid(double x);
};

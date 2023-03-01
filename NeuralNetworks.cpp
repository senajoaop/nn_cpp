#include <iostream>
#include "MLP.h"

int main()
{
  srand(time(NULL));
  rand();

  Perceptron *p = new Perceptron(2);

  p->set_weights({10, 10, -15});

  cout << p->run({0, 0}) << endl;
  cout << p->run({0, 1}) << endl;
  cout << p->run({1, 0}) << endl;
  cout << p->run({1, 1}) << endl;
}

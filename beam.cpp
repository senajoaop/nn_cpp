// #include <iostream>
// #include <vector>
#include "constants.h"
#include "tools.h"

#include <python3.10/Python.h>

// #define MESH 100

struct Beam {
    const int n = 10;
    const int MESH = 200;
    const int size = n * MESH;

    double L;
    double m;
    double YI;

    // decidir quais são os parametros de entrada
    // decidir um novo construtor

    std::vector<double> x;
    std::vector<double> lamb_r;
    std::vector<double> sigma_r;
    std::vector<double> omega_r;
    std::vector<double> phi_r;

    Beam(double beamSize,
         double beamWidth,
         double mass,
         double stiffness,) : x(MESH),
                             lamb_r(n),
                             sigma_r(n),
                             omega_r(n),
                             phi_r(size) {
        L = beamSize;
        m = mass;
        YI = stiffness;

        double step = L / (MESH-1);

        for (int i = 0; i<MESH; i++) {
            x[i] = i * step;
        }

        lamb_r = calculate_lambda();

        for (int i = 0; i<n; i++) {
            sigma_r[i] = (sin(lamb_r[i]) - sinh(lamb_r[i])) / 
                         (cos(lamb_r[i]) + cosh(lamb_r[i]));
        }

        for (int i = 0; i<n; i++) {
            sigma_r[i] = lamb_r[i] * lamb_r[i] * std::pow(YI / (m * std::pow(L, 4.0)), 1.0 / 2.0)
        }
    }

    void calculate_phi() {
        int idx;
        for (int i = 0; i<n; i++) {
            for (int j = 0; j<MESH; j++) {
                idx = i*MESH + j;
                phi_r[idx] = std::pow(1.0 / (m * L), 1.0 / 2.0)
                    * (cos(lamb_r[i]/L*x[j]) - cosh(lamb_r[i]/L*x[j]) +
                      sigma_r[i] * (sin(lamb_r[i]/L*x[j]) - sinh(lamb_r[i]/L*x[j]))); 
            }
        }
    }

    void printv(std::vector<double> vec) {
        int idx;
        for (int i = 0; i<n; i++) {
            for (int j = 0; j<MESH; j++) {
                idx = i*MESH + j;
                if (j==0)
                    std::cout << "["; 
                else if (j==(MESH-1))
                    std::cout << vec[idx] << "]"; 
                else
                    std::cout << vec[idx] << ", "; 
            }
            std::cout << std::endl;
        }
    }
};

int main() {

    Beam bb = Beam(55.5, 0.4);
    bb.lamb_r = calculate_lambda();
    bb.calculate_phi();

    bb.printv(bb.phi_r);

    // for (int i = 0; i<bb.n; i++) {
    //     for (int j = 0; j<bb.MESH; j++) {
    //         std::cout << bb.phi_r[i][j]; 
    //     }
    //     std::endl;
    // }

    // std::cout << "olá " << bb.L << std::endl;
    //
    // plot_vector(bb.x, std::vector<double>(&bb.phi_r[0], &bb.phi_r[bb.MESH]));
    // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESH], &bb.phi_r[bb.MESH*2]));
    // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESH*2], &bb.phi_r[bb.MESH*3]));
    // Py_Initialize();
    // PyRun_SimpleString("from time import time,ctime\n"
    //              "print('Today is',ctime(time()))\n");
    // Py_Finalize();
}

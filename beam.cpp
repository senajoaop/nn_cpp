// #include <iostream>
// #include <vector>
#include "constants.h"
#include "tools.h"

#include <python3.10/Python.h>

typedef std::complex<double> dcomplex;

// #define MESHx 100

struct Beam {
  const int n = 6;
  const int MESHx = 200;
  const int MESHw = 300;
  const int size = n * MESHx;
  const dcomplex _1j = {0.0, 1.0};
  const std::string config = "series";
  // const std::string config = "parallel";

  double L;
  double b; 
  double Ys;
  double c11;
  double e31;
  double eps33;
  double hs;
  double hp;
  double ps;
  double pp;
  double f_i;
  double f_f;
  double R;

  double m;
  double YI;
  double hpc;

  // decidir um novo construtor

  std::vector<double> x;
  std::vector<double> lamb_r;
  std::vector<double> sigma_r;
  std::vector<double> omega_r;
  std::vector<double> phi_r;
  std::vector<double> dphi_r;
  std::vector<double> sigma_mass;
  std::vector<double> theta_r;
  std::vector<double> Cp;
  
  std::vector<double> zeta = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

  std::vector<dcomplex> w;
  std::vector<dcomplex> FRFwrel;

  Beam() : x(MESHx), lamb_r(n), sigma_r(n), omega_r(n), phi_r(size), dphi_r(size), sigma_mass(n), theta_r(n), Cp(n), w(MESHw), FRFwrel(MESHw) {

    properties();

    double stepx = L / (MESHx-1);

    for (int i = 0; i<MESHx; i++) {
      x[i] = i * stepx;
    }

    double stepw = (f_f - f_i) / (MESHw-1);

    for (int i = 0; i<MESHw; i++) {
      w[i] = f_i + i * stepw;
      FRFwrel[i] = 0.0;
    }

    lamb_r = calculate_lambda();

    for (int i = 0; i<n; i++) {
      sigma_r[i] = (sin(lamb_r[i]) - sinh(lamb_r[i])) / 
        (cos(lamb_r[i]) + cosh(lamb_r[i]));
    }

    for (int i = 0; i<n; i++) {
      omega_r[i] = pow(lamb_r[i], 2) * sqrt(YI / (m * pow(L, 4)));
    }
  }

  void properties() {
    std::fstream newfile;
    double data[13];
    newfile.open("input_data.txt",std::ios::in);

    if (newfile.is_open()){
      std::string tp;
      int l = 0;
      while(std::getline(newfile, tp)) {
        data[l] = stod(tp);
        l++;
      }
      newfile.close();
    }

    L = data[0];
    b = data[1]; 
    Ys = data[2];
    c11 = data[3];
    e31 = data[4];
    eps33 = data[5];
    hs = data[6];
    hp = data[7];
    ps = data[8];
    pp = data[9];
    f_i = data[10];
    f_f = data[11];
    R = data[12];

    YI = ((2 * b) / 3) * ((Ys * (pow(hs, 3) / 8)) + c11 * (pow(hp + (hs / 2), 3) - (pow(hs, 3) / 8)));
    m = b * (ps * hs + 2 * pp * hp);
    hpc = hs/2 + hp/2;
  }

  void calculate_phi() {
    int idx;
    for (int i = 0; i<n; i++) {
      for (int j = 0; j<MESHx; j++) {
        idx = i*MESHx + j;

        phi_r[idx] = sqrt(1/(m*L))*(cos(lamb_r[i]/L*x[j]) - cosh(lamb_r[i]/L*x[j]) + sigma_r[i]*(sin(lamb_r[i]/L*x[j]) - sinh(lamb_r[i]/L*x[j]))); 

        dphi_r[idx] = sqrt(1/(L*m))*(sigma_r[i]*(lamb_r[i]*cos(lamb_r[i]*x[j]/L)/L - lamb_r[i]*cosh(lamb_r[i]*x[j]/L)/L) - lamb_r[i]*sin(lamb_r[i]*x[j]/L)/L - lamb_r[i]*sinh(lamb_r[i]*x[j]/L)/L);
      }
    }

    double trapz;
    for (int i = 0; i<n; i++) {
      trapz = 0.0;
      for (int j = 0; j<MESHx; j++) {
        idx = i*MESHx + j;

        if (j!=MESHx-1) {
          trapz += (phi_r[idx] + phi_r[idx+1]) * (x[j+1] - x[j]) / 2;
        }
      }
      sigma_mass[i] = - m * trapz;
    }

    for (int i=0; i<n; i++) {
      idx = MESHx*(i + 1) - 1;
      if (config=="series") {
        theta_r[i] = e31 * b * hpc * dphi_r[idx];
        Cp[i] = (eps33*b*L)/(2*hp);
      } else {
        theta_r[i] = 2 * e31 * b * hpc * dphi_r[idx];
        Cp[i] = (2*eps33*b*L)/(hp);
      }
    }
  }

  void calculate_FRF() {
    dcomplex T1;
    dcomplex T2;
    dcomplex T3;
    dcomplex T4;

    int idx;

    for (int i=0; i<MESHw; i++) {
      FRFwrel[i] = 0.0;
      for (int j=0; j<n; j++) {
        idx = MESHx*(j + 1) - 1;
        T1 = 0.0;
        T3 = 0.0;
        for (int k=0; k<n; k++) {
          T1 += (_1j*w[i]*theta_r[k]*sigma_mass[k])/(pow(omega_r[k], 2) - pow(w[i], 2) + _1j*2.0*zeta[k]*omega_r[k]*w[i]);
          T3 += (_1j*w[i]*pow(theta_r[k], 2))/(pow(omega_r[k], 2) - pow(w[i], 2) + _1j*2.0*zeta[k]*omega_r[k]*w[i]);
        }
      
        T2 = 1/R + _1j*w[i]*Cp[j];
        T4 = sigma_r[idx]/(pow(omega_r[j], 2) - pow(w[i], 2) + _1j*2.0*zeta[j]*omega_r[j]*w[i]);

        FRFwrel[i] += (sigma_mass[j] - theta_r[j]*T1/(T2 + T3))*(T4);
      }
    }
  }

  void printv(std::vector<double> vec) {
    int idx;
    for (int i = 0; i<n; i++) {
      for (int j = 0; j<MESHx; j++) {
        idx = i*MESHx + j;
        if (j==0)
          std::cout << "["; 
        else if (j==(MESHx-1))
            std::cout << vec[idx] << "]"; 
          else
            std::cout << vec[idx] << ", "; 
      }
      std::cout << std::endl;
    }
  }
};

int main() {

  Beam bb = Beam();
  bb.lamb_r = calculate_lambda();
  bb.calculate_phi();

  bb.calculate_FRF();

  std::vector<double> X(bb.MESHw);
  std::vector<double> Y(bb.MESHw);

  for (int i=0; i<bb.MESHw; i++) {
    X[i] = sqrt(pow(real(bb.w[i]), 2) + pow(imag(bb.w[i]), 2));
    Y[i] = sqrt(pow(real(bb.FRFwrel[i]), 2) + pow(imag(bb.FRFwrel[i]), 2));
  }
  plot_vector(X, Y);


  // for (int i = 0; i<bb.n; i++) {
  //   std::cout << bb.sigma_mass[i] << std::endl;
  // }


  // bb.properties();

  // bb.printv(bb.dphi_r);

  // for (int i = 0; i<bb.n; i++) {
  //     for (int j = 0; j<bb.MESHx; j++) {
  //         std::cout << bb.phi_r[i][j]; 
  //     }
  //     std::endl;
  // }

  // std::cout << "olá " << bb.L << std::endl;
  //
  // multiplot_vector(bb.x, bb.phi_r, bb.MESHx, bb.n);
  // plot_vector(bb.x, std::vector<double>(&bb.phi_r[0], &bb.phi_r[bb.MESHx]));
  // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESHx], &bb.phi_r[bb.MESHx*2]));
  // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESHx*2], &bb.phi_r[bb.MESHx*3]));
  // Py_Initialize();
  // PyRun_SimpleString("from time import time,ctime\n"
  //              "print('Today is',ctime(time()))\n");
  // Py_Finalize();
}

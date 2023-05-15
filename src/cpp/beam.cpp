// #include <iostream>
// #include <vector>
#include "constants.h"
#include "tools.h"

// #include <python3.10/Python.h>
// #include <Python.h>

typedef std::complex<double> dcomplex;

// extern Beam bstruc;

// #define MESHx 100

struct Beam {
  const int n = 8;
  const int MESHx = 100;
  const int MESHw = 50000;
  const int MESHw_alt = 800;
  const double nf_alt = 80.0;
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
  double zeta1;
  double zeta2;

  double m;
  double YI;
  double hpc;


  std::vector<double> x;
  std::vector<double> lamb_r;
  std::vector<double> sigma_r;
  std::vector<double> omega_r;
  std::vector<double> phi_r;
  std::vector<double> dphi_r;
  std::vector<double> sigma_mass;
  std::vector<double> theta_r;
  std::vector<double> Cp;
  std::vector<double> damp;
  std::vector<double> f;
  std::vector<double> FRFwrelABS;
  std::vector<double> dFRFwrelABS;
  std::vector<double> f_alt;
  std::vector<double> FRFwrelABS_alt;

  std::vector<double> zeta = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

  std::vector<dcomplex> w;
  std::vector<dcomplex> FRFwrel;
  std::vector<dcomplex> dFRFwrel;
  std::vector<dcomplex> w_alt;
  std::vector<dcomplex> FRFwrel_alt;

  Beam(std::string fileName) : x(MESHx), lamb_r(n), sigma_r(n), omega_r(n), phi_r(size), dphi_r(size), sigma_mass(n), theta_r(n), Cp(n), damp(n), f(MESHw), f_alt(MESHw_alt), w(MESHw), FRFwrel(MESHw), dFRFwrel(MESHw), FRFwrelABS(MESHw), dFRFwrelABS(MESHw), w_alt(MESHw_alt), FRFwrel_alt(MESHw_alt), FRFwrelABS_alt(MESHw_alt) {

    properties(fileName);

    double stepx = L / (MESHx-1);

    for (int i = 0; i<MESHx; i++) {
      x[i] = i * stepx;
    }

    double stepw = (f_f - f_i) / (MESHw-1);

    for (int i = 0; i<MESHw; i++) {
      f[i] = f_i + i * stepw;
      w[i] = f[i] * 2 * M_PI;
      // FRFwrel[i] = 0.0;
      // dFRFwrel[i] = 0.0;
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

  void properties(std::string fileName) {
    std::fstream newfile;
    double data[15];
    newfile.open(fileName,std::ios::in);

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
    zeta1 = data[13];
    zeta2 = data[14];


    // for (int jj=0; jj<15; jj++) std::cout << data[jj] << std::endl;

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

  void calculate_damping() {
    damp[0] = zeta1;
    damp[1] = zeta2;

    double T1 = (2*omega_r[0]*omega_r[1])/(pow(omega_r[0], 2) - pow(omega_r[1], 2));
    std::vector<double> matProp = { YI/omega_r[1], -YI/omega_r[0], -m*omega_r[1], m*omega_r[0] };
    std::vector<double> zeta = { damp[0], damp[1] };
    std::vector<double> csIca(2);

    for (int i=0; i<4; i++) {
      matProp[i] *= T1;
    }

    csIca = matrix_multiplication_linsys(matProp, zeta);

    for (int i=2; i<n; i+=2) {
      T1 = (2*omega_r[i]*omega_r[i+1])/(pow(omega_r[i], 2) - pow(omega_r[i+1], 2));
      matProp[0] = T1*(YI/omega_r[i+1]);
      matProp[1] = T1*(-YI/omega_r[i]);
      matProp[2] = T1*(-m*omega_r[i+1]);
      matProp[3] = T1*(m*omega_r[i]);
      zeta = solve_linear_system(matProp, csIca);
      damp[i] = zeta[0];
      damp[i+1] = zeta[1];
    }
  }

  void calculate_FRF() {
    dcomplex T1, T2, T3, T4, T5;

    int idx;

    for (int i=0; i<MESHw; i++) {
      FRFwrel[i] = 0.0;
      dFRFwrel[i] = 0.0;
      for (int j=0; j<n; j++) {
        idx = MESHx*(j + 1) - 1;
        T1 = 0.0;
        T4 = 0.0;
        for (int k=0; k<n; k++) {
          T1 += (_1j*w[i]*theta_r[k]*sigma_mass[k])/(pow(omega_r[k], 2) - pow(w[i], 2) + _1j*2.0*damp[k]*omega_r[k]*w[i]);
          T4 += (_1j*w[i]*pow(theta_r[k], 2))/(pow(omega_r[k], 2) - pow(w[i], 2) + _1j*2.0*damp[k]*omega_r[k]*w[i]);
        }

        T2 = 1/R;
        T3 = _1j*w[i]*Cp[j];
        T5 = phi_r[idx]/(pow(omega_r[j], 2) - pow(w[i], 2) + _1j*2.0*damp[j]*omega_r[j]*w[i]);

        FRFwrel[i] += (sigma_mass[j] - theta_r[j]*T1/(T2 + T3 + T4))*(T5);
        dFRFwrel[i] += - (theta_r[j]*T1*T5)/(pow(R, 2)*pow(T2 + T3 + T4, 2));

        FRFwrelABS[i] = log10(sqrt(pow(real(FRFwrel[i]), 2) + pow(imag(FRFwrel[i]), 2)));
        dFRFwrelABS[i] = log10(sqrt(pow(real(dFRFwrel[i]), 2) + pow(imag(dFRFwrel[i]), 2)));
      }
    }
  }

  double peak_first_mode() {
    int idx;

    idx = get_idx_peak(FRFwrelABS);

    return FRFwrelABS[idx];
  }

  double peak_first_mode_derivative() {
    int idx;

    idx = get_idx_peak(FRFwrelABS);

    return dFRFwrelABS[idx];
  }

  void update_resistante(double resistance) {
    R = resistance;
  }

  void update_frequency_FRF(double f_peak) {
    double stepw = nf_alt / (MESHw_alt-1);

    for (int i = 0; i<MESHw_alt; i++) {
      f_alt[i] = f_peak - nf_alt/2 + i*stepw;
      w_alt[i] = f_alt[i]*2*M_PI;

      // std::cout << f_alt[i] << std::endl;
    }

    dcomplex T1, T2, T3, T4, T5;

    int idx;

    for (int i=0; i<MESHw_alt; i++) {
      FRFwrel_alt[i] = 0.0;
      // dFRFwrel[i] = 0.0;
      for (int j=0; j<n; j++) {
        idx = MESHx*(j + 1) - 1;
        T1 = 0.0;
        T4 = 0.0;
        for (int k=0; k<n; k++) {
          T1 += (_1j*w_alt[i]*theta_r[k]*sigma_mass[k])/(pow(omega_r[k], 2) - pow(w_alt[i], 2) + _1j*2.0*damp[k]*omega_r[k]*w_alt[i]);
          T4 += (_1j*w_alt[i]*pow(theta_r[k], 2))/(pow(omega_r[k], 2) - pow(w_alt[i], 2) + _1j*2.0*damp[k]*omega_r[k]*w_alt[i]);
        }

        T2 = 1/R;
        T3 = _1j*w_alt[i]*Cp[j];
        T5 = phi_r[idx]/(pow(omega_r[j], 2) - pow(w_alt[i], 2) + _1j*2.0*damp[j]*omega_r[j]*w_alt[i]);

        FRFwrel_alt[i] += (sigma_mass[j] - theta_r[j]*T1/(T2 + T3 + T4))*(T5);
        // dFRFwrel[i] += - (theta_r[j]*T1*T5)/(pow(R, 2)*pow(T2 + T3 + T4, 2));

        FRFwrelABS_alt[i] = log10(sqrt(pow(real(FRFwrel_alt[i]), 2) + pow(imag(FRFwrel_alt[i]), 2)));
        // dFRFwrelABS[i] = log10(sqrt(pow(real(dFRFwrel[i]), 2) + pow(imag(dFRFwrel[i]), 2)));
      }
    }
  }


  void update_properties(std::string fileName) {
    properties(fileName);

    double stepx = L / (MESHx-1);

    for (int i = 0; i<MESHx; i++) {
      x[i] = i * stepx;
    }

    double stepw = (f_f - f_i) / (MESHw-1);

    for (int i = 0; i<MESHw; i++) {
      f[i] = f_i + i * stepw;
      w[i] = f[i] * 2 * M_PI;
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

// int main() {
//
//   Beam bb = Beam();
//   bb.lamb_r = calculate_lambda();
//   bb.calculate_phi();
//   bb.calculate_damping();
//
//   std::cout << bb.R << std::endl;
//
//   bb.calculate_FRF();
//
//   std::vector<double> X(bb.MESHw);
//   std::vector<double> Y(bb.MESHw);
//
//   for (int i=0; i<bb.MESHw; i++) {
//     X[i] = sqrt(pow(real(bb.w[i]), 2) + pow(imag(bb.w[i]), 2));
//     Y[i] = log10(sqrt(pow(real(bb.FRFwrel[i]), 2) + pow(imag(bb.FRFwrel[i]), 2)));
//     // Y[i] = log10(sqrt(pow(real(bb.dFRFwrel[i]), 2) + pow(imag(bb.dFRFwrel[i]), 2)));
//   }
//   int idx = get_idx_peak(Y);
//   std::cout << idx << ", " << bb.f[idx] << std::endl;
//   // plot_vector(bb.f, Y);
//
//
//   bb.update_resistante(0.0000001);
//   std::cout << bb.R << std::endl;
//   bb.calculate_FRF();
//
//   // std::vector<double> X(bb.MESHw);
//   // std::vector<double> Y(bb.MESHw);
//
//   for (int i=0; i<bb.MESHw; i++) {
//     X[i] = sqrt(pow(real(bb.w[i]), 2) + pow(imag(bb.w[i]), 2));
//     Y[i] = log10(sqrt(pow(real(bb.FRFwrel[i]), 2) + pow(imag(bb.FRFwrel[i]), 2)));
//     // Y[i] = log10(sqrt(pow(real(bb.dFRFwrel[i]), 2) + pow(imag(bb.dFRFwrel[i]), 2)));
//   }
//   idx = get_idx_peak(Y);
//   std::cout << idx << ", " << bb.f[idx] << std::endl;
//   // plot_vector(bb.f, Y);
//
//   // std::cout << std::endl; 
//   // for (int i=0; i<bb.n; i++) {
//   //   std::cout << bb.omega_r[i] / 2 / M_PI << std::endl;
//   // }
//
//   // solve_linear_system(;
//
//
//   // for (int i = 0; i<bb.n; i++) {
//   //   std::cout << bb.sigma_mass[i] << std::endl;
//   // }
//
//
//   // bb.properties();
//
//   // bb.printv(bb.dphi_r);
//
//   // for (int i = 0; i<bb.n; i++) {
//   //     for (int j = 0; j<bb.MESHx; j++) {
//   //         std::cout << bb.phi_r[i][j]; 
//   //     }
//   //     std::endl;
//   // }
//
//   // std::cout << "olá " << bb.L << std::endl;
//   //
//   // multiplot_vector(bb.x, bb.phi_r, bb.MESHx, bb.n);
//   // plot_vector(bb.x, std::vector<double>(&bb.phi_r[0], &bb.phi_r[bb.MESHx]));
//   // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESHx], &bb.phi_r[bb.MESHx*2]));
//   // plot_vector(bb.x, std::vector<double>(&bb.phi_r[bb.MESHx*2], &bb.phi_r[bb.MESHx*3]));
//   // Py_Initialize();
//   // PyRun_SimpleString("from time import time,ctime\n"
//   //              "print('Today is',ctime(time()))\n");
//   // Py_Finalize();
// }




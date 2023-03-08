// #include <python3.8/Python.h>
#include <Python.h>
#include <string.h>

void plot_vector(std::vector<double> X, std::vector<double> Y) {
  Py_Initialize();

  std::string x = "";
  std::string y = "";

  x += "[";
  y += "[";
  for (int i = 0; i<X.size(); i++) {
    x += std::to_string(X[i]);
    y += std::to_string(Y[i]);
    if (i!=X.size()-1) {
      x += ",";
      y += ",";
    }
  }
  x += "]";
  y += "]";

  // std::cout << x << std::endl;

  std::string script = "import matplotlib.pyplot as plt\n"
    "plt.plot("+x+","+y+")\n"
    "plt.show()\n";

  // std::cout << script << std::endl;
  PyRun_SimpleString((script).c_str());
  Py_Finalize();

  std::ofstream out("script_plot.py");
  out << script;
  out.close();
}


void multiplot_vector(std::vector<double> X, std::vector<double> Y, int nv, int nplt) {
  Py_Initialize();

  std::string script2 = "";
  int idx;

  for (int j=0; j<nplt; j++) {
    std::string x = "";
    std::string y = "";

    x += "[";
    y += "[";
    for (int i = 0; i<X.size(); i++) {
      idx = j * X.size() + i;

      x += std::to_string(X[i]);
      y += std::to_string(Y[idx]);
      if (i!=X.size()-1) {
        x += ",";
        y += ",";
      }
    }
    x += "]";
    y += "]";

    script2 += "plt.plot("+x+","+y+",label=\"Modo "+std::to_string(j)+"\")\n";
  }

  std::string script1 = "import matplotlib.pyplot as plt\n";
  std::string script3 = "plt.legend()\nplt.show()\n";

  std::string script = script1 + script2 + script3;

  // std::cout << script << std::endl;
  PyRun_SimpleString((script).c_str());
  Py_Finalize();
}


std::vector<double> matrix_multiplication_linsys(std::vector<double> A, std::vector<double> B) {

  int idx1, idx2, idx3;
  std::vector<double> res(2);

  for(int i=0; i<2; ++i) {
    for(int j=0; j<1; ++j) {
      for(int k=0; k<2; ++k) {
        idx1 = i*1 + j;
        idx2 = i*2 + k;
        idx3 = k*1 + j;
        res[idx1] += A[idx2] * B[idx3];
      }
    }
  }
  
  return res;
}

std::vector<double> solve_linear_system(std::vector<double> A, std::vector<double> B) {
  int i,j,k,n;

  std::vector<double> resvec(2);

  double mat[2][3] = { { A[0], A[1], B[0] }, { A[2], A[3], B[1] } };

  double res[2];

  n = 2;

  // cout<<"\nEnter the elements of the augmented matrix: ";
  // for(i=0;i<n;i++)
  // {
  //   for(j=0;j<n+1;j++)
  // {
  //   cin>>mat[i][j]; 
  // }    
  // }

  for(i=0; i<n; i++) {                   
      for(j=i+1; j<n; j++) {
          if(abs(mat[i][i]) < abs(mat[j][i])) {
            for(k=0; k<n+1; k++) {
                mat[i][k] = mat[i][k] + mat[j][k];
                mat[j][k] = mat[i][k] - mat[j][k];
                mat[i][k] = mat[i][k] - mat[j][k];
            }
          }
      }
  }

  for(i=0; i<n-1; i++) {
      for(j=i+1; j<n; j++) {
          float f=mat[j][i]/mat[i][i];
          for(k=0;k<n+1;k++) {
              mat[j][k]=mat[j][k]-f*mat[i][k];
          }
      }
  }

  for(i=n-1; i>=0; i--) {                     
      res[i]=mat[i][n];

      for(j=i+1; j<n; j++) {
          if(i!=j) {
            res[i]=res[i]-mat[i][j]*res[j];
          }          
        }
      res[i]=res[i]/mat[i][i];  
  }

  // std::cout<<"\nThe values of unknowns for the above equations=>\n";
  // for(i=0; i<n; i++) {
  //   std::cout<<res[i]<<"\n";
  // }

  resvec[0] = res[0];
  resvec[1] = res[1];

  return resvec;
}

int get_idx_peak(std::vector<double> vec) {
  if (vec.size() == 1)
    return 0;
  // if (vec[0] >= vec[1])
  //   return 0;
  // if (vec[vec.size() - 1] >= vec[vec.size() - 2])
  //   return vec.size() - 1;

  for (int i=1; i<vec.size()-1; i++) {
    if (vec[i] >= vec[i-1] && vec[i] >= vec[i+1])
      return i;
  }

  return 0;
}

std::vector<int> get_idx_peaks(std::vector<double> vec) {
  std::vector<int> peaks;

  if (vec.size() == 1) {
    peaks.push_back(0);
    return peaks;
  }

  // if (vec[vec.size() - 1] >= vec[vec.size() - 2]) {
  //   peaks.push_back(vec.size() - 1);
  //   return peaks;
  // }

  for (int i=1; i<vec.size()-1; i++) {
    if (vec[i] >= vec[i-1] && vec[i] >= vec[i+1])
      peaks.push_back(i);
  }

  return peaks;
}


void random_data_generator(int n) {

  double L, b, Ys, c11, e31, eps33, hs, hp, ps, pp;

	srand((unsigned) time(NULL));

  std::ofstream fout("random_data.txt");
  fout << std::setprecision(20);
  fout << "L,b,Ys,c11,e31,eps33,hs,hp,ps,pp";

  for (int i=0; i<n; i++) {
    L = 10.0e-3 + (double)rand()/RAND_MAX*(100.0e-3 - 10.0e-3);
    b = 2.0e-3 + (double)rand()/RAND_MAX*(10.0e-3 - 2.0e-3);
    Ys = 10.0e8 + (double)rand()/RAND_MAX*(10.0e10 - 10.0e8);
    c11 = 10.0e8 + (double)rand()/RAND_MAX*(10.0e10 - 10.0e8);
    e31 = -1.0 + (double)rand()/RAND_MAX*(-20.0 - (-1.0));
    eps33 = 1.0e-9 + (double)rand()/RAND_MAX*(20.0e-9 - 1.0e-9);
    hs = 0.1e-3 + (double)rand()/RAND_MAX*(1.0e-3 - 0.1e-3);
    hp = 0.1e-3 + (double)rand()/RAND_MAX*(1.0e-3 - 0.1e-3);
    ps = 1.0e3 + (double)rand()/RAND_MAX*(5.0e3 - 1.0e3);
    pp = 5.0e3 + (double)rand()/RAND_MAX*(10.0e3 - 5.0e3);

    fout << '\n';
    fout << L << ',' << b << ',' << Ys << ',' << c11 << ',' << e31 << ',' << eps33 << ',' << hs << ',' << hp << ',' << ps << ',' << pp;
    // std::cout << L << ',' << b << ',' << Ys << ',' << c11 << ',' << e31 << ',' << eps33 << ',' << hs << ',' << hp << ',' << ps << ',' << pp << std::endl;
  }

  fout.close();

}

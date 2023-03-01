#include <python3.10/Python.h>
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

    std::cout << x << std::endl;

    std::string script = "import matplotlib.pyplot as plt\n"
                 "plt.plot("+x+","+y+")\n"
                 "plt.show()\n";

    std::cout << script << std::endl;
    PyRun_SimpleString((script).c_str());
    Py_Finalize();
}


void teste() {
    // Py_SetProgramName(argv[0]);  /* optional but recommended */
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                 "print('Today is',ctime(time()))\n");
    Py_Finalize();
}

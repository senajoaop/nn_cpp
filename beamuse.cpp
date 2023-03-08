#include "beam.cpp"
#include <stdio.h>



int main() {

  // double x = 0.1;
  //
  // char* str = "0X1.999999999999AP-4";
  //
  // printf("0.1 in hexadecimal is: %A\n", x);
  //
  // printf("Now reading %s\n", str);
  //
  // /* in a production code I would check for errors */
  // sscanf(str, "%lA", &x); /* note: %lA is used! */
  //
  // printf("It equals %g\n", x);
  //


 //  std::vector<std::vector<std::string>> data;
 //  std::ifstream infile("res_opt.txt");
	//
 //  while (infile) {
 //      std::string s;
 //      if (!getline(infile, s)) break;
	//
 //      std::istringstream ss(s);
 //      std::vector<std::string> record;
	//
 //      while (ss) {
 //          std::string s;
 //          if (!getline(ss, s, ',')) break;
 //          record.push_back(s);
 //        }
	//
 //      data.push_back(record);
 //  }
	//
 //  infile.close();
	//
	//
 //  
 //  for (std::vector<std::string> item : data) {
 //    for (std::string i : item)
 //      std::cout << i << ',';
 //    std::cout << std::endl;
 //  }
	//
	//
 //  // Providing a seed value
	// srand((unsigned) time(NULL));
	//
	// // Get a random number
	// // int random = rand();
 //  double f = (double)rand() / RAND_MAX;
 //  double g = 100 + f * (250 - 100);
	//
	//
	// // Print the random number
 //  std::cout << random << std::endl;
 //  std::cout << f << std::endl;
 //  std::cout << g << std::endl;
	//
  random_data_generator(3);


  // Beam bb = Beam("input_data.txt");
  // bb.lamb_r = calculate_lambda();
  // bb.calculate_phi();
  // bb.calculate_damping();
  //
  // std::cout << bb.R << std::endl;
  //
  // bb.calculate_FRF();
  //
  // std::vector<double> X(bb.MESHw);
  // std::vector<double> Y(bb.MESHw);
  //
  // std::vector<int> pp;
  //
  // for (int i=0; i<bb.MESHw; i++) {
  //   X[i] = sqrt(pow(real(bb.w[i]), 2) + pow(imag(bb.w[i]), 2));
  //   Y[i] = log10(sqrt(pow(real(bb.FRFwrel[i]), 2) + pow(imag(bb.FRFwrel[i]), 2)));
  // }
  // int idx = get_idx_peak(Y);
  // pp = get_idx_peaks(Y);
  // std::cout << idx << ", " << bb.f[idx] << std::endl;
  // for (int i=0; i<pp.size(); i++)
  //   std::cout << pp[i] << ", " << bb.f[pp[i]] << std::endl;
  // // plot_vector(bb.f, Y);
  //
  // bb.update_frequency_FRF(bb.f[idx]);
  // plot_vector(bb.f_alt, bb.FRFwrelABS_alt);
  //
  // std::ofstream fout("data_vec.txt");
  // fout << std::setprecision(10);
  //
  // for(auto const& DAT : bb.f_alt)
  //       fout << DAT << ',';
  //
  //
  // for (int y : pp)
  //   std::cout << "resultado: " << y << std::endl;
  //
  // for (double i=-20; i<20; i++) {
  //
  //   std::cout << pow(10, i/2) << std::endl;
  //
  //   bb.update_resistante(pow(10, i/2));
  //   // bb.calculate_FRF();
  //   bb.update_frequency_FRF(bb.f[pp[2]]);
  //
  //   fout << '\n';
  //   for(auto const& DAT : bb.FRFwrelABS_alt)
  //       fout << DAT << ',';
  // }
  //
  // fout.close();
  //
  //
  //
  // bb.update_resistante(0.0000001);
  // std::cout << bb.R << std::endl;
  // bb.calculate_FRF();
  //
  //
  // for (int i=0; i<bb.MESHw; i++) {
  //   X[i] = sqrt(pow(real(bb.w[i]), 2) + pow(imag(bb.w[i]), 2));
  //   Y[i] = log10(sqrt(pow(real(bb.FRFwrel[i]), 2) + pow(imag(bb.FRFwrel[i]), 2)));
  // }
  // idx = get_idx_peak(Y);
  // pp = get_idx_peaks(Y);
  // std::cout << idx << ", " << bb.f[idx] << std::endl;
  // for (int i=0; i<pp.size(); i++)
  //   std::cout << pp[i] << ", " << bb.f[pp[i]] << std::endl;
  // // plot_vector(bb.f, Y);
  //
  // for (int i=0; i<pp.size(); i++) {
  //   bb.update_frequency_FRF(bb.f[pp[i]]);
  //   std::cout << pp[i] << ", " << bb.f[pp[i]] << std::endl;
  // }
}

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include "beam.cpp"





// extern Beam bstruc;
Beam bstruc = Beam("/../data/input_data.txt");
int mode;
int idxMode;
int idx;




double my_f (const gsl_vector *v, void *params) {
  double R;
  (void)(params);
  // double *p = (double *)params;

  R = gsl_vector_get(v, 0);
  // x = gsl_vector_get(v, 0);
  // y = gsl_vector_get(v, 1);

  // bstruc = Beam()
  bstruc.update_resistante(R);
  // bstruc.calculate_FRF();
  bstruc.update_frequency_FRF(bstruc.f[idxMode]);
  idx = get_idx_peak(bstruc.FRFwrelABS_alt);

  // return bstruc.peak_first_mode();
  return bstruc.FRFwrelABS_alt[idx];
}

void my_df (const gsl_vector *v, void *params, gsl_vector *df) {
  double R;
  (void)(params);
  // double *p = (double *)params;

  R = gsl_vector_get(v, 0);
  // x = gsl_vector_get(v, 0);
  // y = gsl_vector_get(v, 1);

  bstruc.update_resistante(R);
  bstruc.calculate_FRF();

  gsl_vector_set(df, 0, bstruc.peak_first_mode_derivative());
  // gsl_vector_set(df, 0, 2.0 * p[2] * (x - p[0]));
  // gsl_vector_set(df, 1, 2.0 * p[3] * (y - p[1]));
}

void my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df) {
  *f = my_f(x, params);
  my_df(x, params, df);
}


void parser_file_input(std::vector<std::string> data) {
  std::ofstream fout("input_opt.txt");
  fout << std::setprecision(20);

  for (int i=0; i<data.size(); i++) {
    if (i != 0) fout << '\n';
    fout << data[i];
  }

  fout << '\n' << bstruc.f_i << '\n' << bstruc.f_f << '\n' << bstruc.R << '\n' << bstruc.zeta1 << '\n' << bstruc.zeta2;

  fout.close();
}


int main (void) {

  random_data_generator(8000);

  int refcalc;
  std::ofstream fout("/../data/res_opt.txt");
  fout << std::setprecision(20);
  fout << "refcalc,nmodes,{iter,R,f,size}*nmodes,L,b,Ys,c11,e31,eps33,hs,hp,ps,pp";
  
  std::vector<std::vector<std::string>> data;
  std::ifstream infile("/../data/random_data.txt");

  while (infile) {
      std::string s;
      if (!getline(infile, s)) break;

      std::istringstream ss(s);
      std::vector<std::string> record;

      while (ss) {
          std::string s;
          if (!getline(ss, s, ',')) break;
          record.push_back(s);
        }

      data.push_back(record);
  }

  infile.close();

  for (int k=1; k<data.size(); k++) {
  refcalc = k;

  size_t iter = 0;
  int status;
  double size;

  parser_file_input(data[k]);

  // bstruc = new Beam("input_opt.txt");
  // mode = 1;

  // bstruc = Beam();
  bstruc.update_properties("/../data/input_opt.txt");
  std::vector<int> peaks;
  bstruc.lamb_r = calculate_lambda();
  bstruc.calculate_phi();
  bstruc.calculate_damping();
  bstruc.calculate_FRF();

  peaks = get_idx_peaks(bstruc.FRFwrelABS);
  // idxMode = bstruc.f[peaks[mode-1]];

  fout << '\n' << refcalc << ',' << peaks.size();

  for (int i=0; i<peaks.size(); i++) {
    iter = 0;

    mode = i+1;
    idxMode = peaks[mode-1];

    // const gsl_multimin_fdfminimizer_type *T;
    // gsl_multimin_fdfminimizer *s;

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *ss, *x;

    // double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };

    // gsl_vector *x;
    // gsl_multimin_function_fdf my_func;
    gsl_multimin_function minex_func;

    // my_func.n = 1;
    // my_func.f = my_f;
    // my_func.df = my_df;
    // my_func.fdf = my_fdf;
    // my_func.params = par;
    minex_func.n = 1;
    minex_func.f = my_f;

    x = gsl_vector_alloc (1);
    gsl_vector_set (x, 0, 1.0);
    // gsl_vector_set (x, 0, 5.0);
    // gsl_vector_set (x, 1, 7.0);

    ss = gsl_vector_alloc (1);
    gsl_vector_set_all (ss, 1.0);

    // T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_vector_bfgs2;
    // s = gsl_multimin_fdfminimizer_alloc (T, 1);
    s = gsl_multimin_fminimizer_alloc (T, 1);

    // gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4);
    gsl_multimin_fminimizer_set (s, &minex_func, x, ss);


    // std::cout << "Passou aqui" << std::endl;

    // do {
    //   iter++;
    //   status = gsl_multimin_fdfminimizer_iterate (s);
    //
    //   if (status)
    //     break;
    //
    //   status = gsl_multimin_test_gradient (s->gradient, 1e-3);
    //
    //   if (status == GSL_SUCCESS)
    //     printf ("Minimum found at:\n");
    //
    //   std::cout << "Iteração" << std::endl;
    //
    //   printf ("%5ld %.5f %10.5f\n", iter,
    //           gsl_vector_get (s->x, 0),
    //           // gsl_vector_get (s->x, 1),
    //           s->f);
    //
    // } while (status == GSL_CONTINUE && iter < 100);
    do {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);

      if (status)
        break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-2);

      if (status == GSL_SUCCESS) {
        printf ("converged to minimum at\n");
        fout << ',' << iter << ',' << gsl_vector_get(s->x, 0) << ',' << s->fval << ',' << size;
        std::cout << "refcalc = " << refcalc << std::endl;
      }

      // printf ("%5ld %10.3e f() = %7.3f size = %.3f\n",
      //         iter,
      //         gsl_vector_get (s->x, 0),
      //         s->fval, size);
    } while (status == GSL_CONTINUE && iter < 100);


    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free (s);

  }

  fout << ',' << bstruc.L << ',' << bstruc.b << ',' << bstruc.Ys << ',' << bstruc.c11 << ',' << bstruc.e31 << ',' << bstruc.eps33 << ',' << bstruc.hs << ',' << bstruc.hp << ',' << bstruc.ps << ',' << bstruc.pp;

  }
  //
  // {}
  //
  fout.close();

  // gsl_multimin_fdfminimizer_free (s);
  // gsl_vector_free (x);

  return 0;
}




















// double
// // my_f (const gsl_vector *v, void *params)
// my_f (const gsl_vector *v, void *params)
// {
//   double x, y;
//   (void)(params);
//   // double *p = (double *)params;
//
//   R = gsl_vector_get(v, 0);
//   // x = gsl_vector_get(v, 0);
//   // y = gsl_vector_get(v, 1);
//
//   return p[2] * (x - p[0]) * (x - p[0]) +
//            p[3] * (y - p[1]) * (y - p[1]) + p[4];
// }
//
// /* The gradient of f, df = (df/dx, df/dy). */
// void
// my_df (const gsl_vector *v, void *params,
//        gsl_vector *df)
// {
//   double x, y;
//   double *p = (double *)params;
//
//   x = gsl_vector_get(v, 0);
//   y = gsl_vector_get(v, 1);
//
//   gsl_vector_set(df, 0, 2.0 * p[2] * (x - p[0]));
//   gsl_vector_set(df, 1, 2.0 * p[3] * (y - p[1]));
// }
//
// /* Compute both f and df together. */
// void
// my_fdf (const gsl_vector *x, void *params,
//         double *f, gsl_vector *df)
// {
//   *f = my_f(x, params);
//   my_df(x, params, df);
// }
//
// int
// main (void)
// {
//   size_t iter = 0;
//   int status;
//
//   const gsl_multimin_fdfminimizer_type *T;
//   gsl_multimin_fdfminimizer *s;
//
//   /* Position of the minimum (1,2), scale factors
//      10,20, height 30. */
//   double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };
//
//   gsl_vector *x;
//   gsl_multimin_function_fdf my_func;
//
//   my_func.n = 2;
//   my_func.f = my_f;
//   my_func.df = my_df;
//   my_func.fdf = my_fdf;
//   my_func.params = par;
//
//   /* Starting point, x = (5,7) */
//   x = gsl_vector_alloc (2);
//   gsl_vector_set (x, 0, 5.0);
//   gsl_vector_set (x, 1, 7.0);
//
//   // T = gsl_multimin_fdfminimizer_conjugate_fr;
//   T = gsl_multimin_fdfminimizer_vector_bfgs2;
//   s = gsl_multimin_fdfminimizer_alloc (T, 2);
//
//   gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4);
//
//   do
//     {
//       iter++;
//       status = gsl_multimin_fdfminimizer_iterate (s);
//
//       if (status)
//         break;
//
//       status = gsl_multimin_test_gradient (s->gradient, 1e-3);
//
//       if (status == GSL_SUCCESS)
//         printf ("Minimum found at:\n");
//
//       printf ("%5ld %.5f %.5f %10.5f\n", iter,
//               gsl_vector_get (s->x, 0),
//               gsl_vector_get (s->x, 1),
//               s->f);
//
//     }
//   while (status == GSL_CONTINUE && iter < 100);
//
//   gsl_multimin_fdfminimizer_free (s);
//   gsl_vector_free (x);
//
//   return 0;
// }
//



























// double fn1 (double x, void * params)
// {
//   (void)(params); /* avoid unused parameter warning */
//   return cos(x) + 1.0;
// }
//
// int
// main (void)
// {
//   int status;
//   int iter = 0, max_iter = 100;
//   const gsl_min_fminimizer_type *T;
//   gsl_min_fminimizer *s;
//   double m = 2.0, m_expected = M_PI;
//   double a = 0.0, b = 6.0;
//   gsl_function F;
//
//   F.function = &fn1;
//   F.params = 0;
//
//   T = gsl_min_fminimizer_brent;
//   s = gsl_min_fminimizer_alloc (T);
//   gsl_min_fminimizer_set (s, &F, m, a, b);
//
//   printf ("using %s method\n",
//           gsl_min_fminimizer_name (s));
//
//   printf ("%5s [%9s, %9s] %9s %10s %9s\n",
//           "iter", "lower", "upper", "min",
//           "err", "err(est)");
//
//   printf ("%5d [%.7f, %.7f] %.7f %+.7f %.7f\n",
//           iter, a, b,
//           m, m - m_expected, b - a);
//
//   do
//     {
//       iter++;
//       status = gsl_min_fminimizer_iterate (s);
//
//       m = gsl_min_fminimizer_x_minimum (s);
//       a = gsl_min_fminimizer_x_lower (s);
//       b = gsl_min_fminimizer_x_upper (s);
//
//       status
//         = gsl_min_test_interval (a, b, 0.001, 0.0);
//
//       if (status == GSL_SUCCESS)
//         printf ("Converged:\n");
//
//       printf ("%5d [%.7f, %.7f] "
//               "%.7f %+.7f %.7f\n",
//               iter, a, b,
//               m, m - m_expected, b - a);
//     }
//   while (status == GSL_CONTINUE && iter < max_iter);
//
//   gsl_min_fminimizer_free (s);
//
//   return status;
// }

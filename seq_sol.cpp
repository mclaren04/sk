#include <cmath>
#include <iostream>
#include <ctime>

// Number of points in the grid
enum {N = 160};

// Initial task: -delta(u(x, y)) + q(x, y)*u(x, y) = F(x, y)
// where x, y belongs to square that has a (A1, B1) point as left down corner
// q and F are khown (note that q = 1 in the square)
// All the side conditions are of the 2nd type
//
// The general task is reduces to the system of
// the linear equations Aw = B
// h1 and h2 - grid steps in x and y directions respectively
// let w - estimated values of the initial function
// B - right part of the abovementioned system
// Ar - matrix multiplication of the right part and matrix r
// F - grid values of function F
// psi_t, psi_r, etc.. are values of psi function from the side conditions
// a and b are the functions from the task
// r = Aw - B is a residual
const double h1 = (double) 5 / (double) (N - 1);
const double h2 = (double) 5 / (double) (N - 1);
const double A1 = -2;
const double B1 = -1;
double **w = 0;
double **F = 0;
double **B = 0;
double **Ar = 0;
double **r = 0;
double **a = 0;
double **b = 0;
double *psi_b = 0;
double *psi_t = 0;
double *psi_l = 0;
double *psi_r = 0;

// m1 = m1 - m2
// m1 and m2 are matrices (size1 * size2)
void matrix_sub(double **m1, double **m2) {
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         m1[i][j] -= m2[i][j];
}

// multiply each element of matrix by scalar
void matrix_mult_by_scalar(double **m, double scalar) {
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         m[i][j] *= scalar;
}

// calculate grid point x_i
double x(int i) {
   return A1 + i*h1;
}

// same for the y
double y(int j) {
   return B1 + j*h2;
}

// calculate F in grid point (x_i, y_j)
double calc_F(int i, int j) {
   double den, den2, den3, nom1, nom2;
   double x_i = x(i);
   double y_j = y(j);
   nom1 = pow(x_i, 4) + pow(y_j, 4) + 2*pow(x_i, 2)*pow(y_j, 2);
   nom2 = 6*(pow(x_i, 2) + pow(y_j, 2)) + 16*x_i*y_j + 5;
   den = 1 + pow(x_i, 2) + pow(y_j, 2);
   return 2*(nom1 + nom2) / pow(den, 3);
}

// k - function from the task
double k(double x, double y) {
   return pow(x + y, 2) + 1;
}

double calc_a(int i, int j) {
   return k(x(i) - 0.5*h1, y(j));
}

double calc_b(int i, int j) {
   return k(x(i), y(j) - 0.5*h2);
}

double calc_psi(int i, int j) {
   double den, nom, mult, x_i, y_j, side;
   x_i = x(i);
   y_j = y(j);
   den = pow(pow(x_i, 2) + pow(y_j, 2) + 1, 2);
   nom = 4 * k(x_i, y_j);
   mult = nom / den;
   if (i == 0)
      side = x_i;
   else if (i == N - 1)
      side = -x_i;
   else if (j == 0)
      side = y_j;
   else if (j == N - 1)
      side = -y_j;
   return side * mult;
}

// rho_{i, j} - function from the definition of scalar product
double rho(int i, int j) {
   double rho1, rho2;
   rho2 = (j == 0 || j == N - 1) ? 0.5 : 1; 
   rho1 = (i == 0 || i == N - 1) ? 0.5 : 1; 
   return rho1 * rho2;
}

double scalar_product(double **u, double **v) {
   double res_aux, res = 0;
   for (int i = 0; i < N; i++) {
      res_aux = 0;
      for (int j = 0; j < N; j++) {
         if (!(i == 0 && j == 0 || i == 0 && j == N-1 ||
                  i == N-1 && j == 0 || i == N-1 && j == N-1))
            res_aux += rho(i, j)*u[i][j]*v[i][j];
      }
      res += res_aux;
   }
   res *= h1*h2;
   return res;
}

double norm(double **u) {
   return sqrt(scalar_product(u, u));
}

double **allocate_matrix(int dim1, int dim2) {
   double **matrix = new double*[dim1];
   for (int i = 0; i < dim1; ++i)
      matrix[i] = new double[dim2];
   return matrix;
}

// numeric derivitive of mat in the direcction x
// in (i, j) node
double numeric_der_x(double **mat, int i, int j) {
   return (mat[i][j] - mat[i - 1][j]) / h1;
}

double numeric_der_y(double **mat, int i, int j) {
   return (mat[i][j] - mat[i][j - 1]) / h2;
}

// calculate value a*w_{x_derivative} in point (i, j)
double numeric_mlt_x(double **mat, int i, int j) {
   return a[i][j] * numeric_der_x(mat, i, j);
}

// same but for y
double numeric_mlt_y(double **mat, int i, int j) {
   return b[i][j] * numeric_der_y(mat, i, j);
}

// value of ( a*w_{x_derivative} )_{x, ij}
double complex_drv_x(double **mat, int i, int j) {
   double summand1, summand2;
   summand1 = a[i + 1][j] * numeric_der_x(mat, i + 1, j);
   summand2 = a[i][j] * numeric_der_x(mat, i, j);
   return (summand1 - summand2) / h1;
}

// value of ( b*w_{y_derivative} )_{y, ij}
double complex_drv_y(double **mat, int i, int j) {
   double summand1, summand2;
   summand1 = b[i][j + 1] * numeric_der_y(mat, i, j + 1);
   summand2 = b[i][j] * numeric_der_y(mat, i, j);
   return (summand1 - summand2) / h2;
}

// calculate Laplace operator
double delta(double **mat, int i, int j) {
   return complex_drv_x(mat, i, j) + complex_drv_y(mat, i, j);
}

// calculate the left part of the equation for
// side condition on the top side
double top_condition_left(double **mat, int i) {
   double summand1 = (2/h1) * numeric_mlt_y(mat, i, N - 1);
   double summand2 = complex_drv_x(mat, i, N - 1);
   return summand1 + mat[i][N - 1] - summand2;
}

double bottom_condition_left(double **mat, int i) {
   double summand1 = -(2/h1) * numeric_mlt_y(mat, i, 1);
   double summand2 = complex_drv_x(mat, i, 0);
   return summand1 + mat[i][0] - summand2;
}

double left_condition_left(double **mat, int j) {
   double summand1 = -(2/h1) * numeric_mlt_x(mat, 1, j);
   double summand2 = complex_drv_y(mat, 0, j);
   return summand1 + mat[0][j] - summand2;
}

double right_condition_left(double **mat, int j) {
   double summand1 = (2/h1) * numeric_mlt_x(mat, N - 1, j);
   double summand2 = complex_drv_y(mat, N - 1, j);
   return summand1 + mat[N - 1][j] - summand2;
}

// common function for calculation of the right
// side in equation for side conditions
double side_condition_right(int i, int j) {
   double ps;
   if (j == N - 1)
      ps = psi_t[i];
   else if (j == 0)
      ps = psi_b[i];
   else if (i == 0)
      ps = psi_l[j];
   else
      ps = psi_r[j];
   return F[i][j] + 2 * ps / h1;
}

// calculate the left part for corner points
double corner_l_b_left(double **mat) {
   return (-2) * numeric_mlt_x(mat, 1, 0) / h1 -
      2 * numeric_mlt_y(mat, 0, 1) / h2 + mat[0][0];
}

double corner_r_b_left(double **mat) {
   return 2 * numeric_mlt_x(mat, N - 1, 0) / h1 -
      2 * numeric_mlt_y(mat, N - 1, 1) / h2 + mat[N - 1][0];
}

double corner_r_t_left(double **mat) {
   return 2 * numeric_mlt_x(mat, N - 1, N - 1) / h1 +
      2 * numeric_mlt_y(mat, N - 1, N - 1) / h2 + mat[N - 1][N - 1];
}

double corner_l_t_left(double **mat) {
   return (-2) * numeric_mlt_x(mat, 1, N - 1) / h1 +
      2 * numeric_mlt_y(mat, 0, N - 1) / h2 + mat[0][N - 1];
}

double calculate_corner_left(double **mat, int i, int j) {
   if (i == 0 && j == 0) {
      return corner_l_b_left(mat);
   } else if (i == N - 1 && j == 0) {
      return corner_r_b_left(mat);
   } else if (i == N - 1 && j == N - 1) {
      return corner_r_t_left(mat);
   } else if (i == 0 && j == N - 1) {
      return corner_l_t_left(mat);
   }
}

double calculate_corner_right(int i, int j) {
   double summand = 4;
   if (j == 0)
      summand *= psi_b[i];
   else 
      summand *= psi_t[i];
   return F[i][j] + summand / h1;
}

// multiply A matrix by mat matrix
void calculate_Amat(double **res, double **mat) {
   for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++)
         res[i][j] = -delta(mat, i, j) + mat[i][j];
   }
   for (int i = 1; i < N - 1; i++) {
      res[i][0] = bottom_condition_left(mat, i);
      res[i][N - 1] = top_condition_left(mat, i);
      res[0][i] = left_condition_left(mat, i);
      res[N - 1][i] = right_condition_left(mat, i);
   }
   res[0][0] = calculate_corner_left(mat, 0, 0);
   res[0][N - 1] = calculate_corner_left(mat, 0, N - 1);
   res[N - 1][0] = calculate_corner_left(mat, N - 1, 0);
   res[N - 1][N - 1] = calculate_corner_left(mat, N - 1, N - 1);
}

// calculate the right part of the numeric task
void calculate_B(double **res) {
   for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
         res[i][j] = F[i][j];
   for (int i = 1; i < N - 1; i++) {
      res[i][0] = side_condition_right(i, 0);
      res[i][N - 1] = side_condition_right(i, N - 1);
      res[0][i] = side_condition_right(0, i);
      res[N - 1][i] = side_condition_right(N - 1, i);
   }
   res[0][0] = calculate_corner_right(0, 0);
   res[0][N - 1] = calculate_corner_right(0, N - 1);
   res[N - 1][0] = calculate_corner_right(N - 1, 0);
   res[N - 1][N - 1] = calculate_corner_right(N - 1, N - 1);
}

void init() {
   w = allocate_matrix(N, N);
   F = allocate_matrix(N, N);
   B = allocate_matrix(N, N);
   Ar = allocate_matrix(N, N);
   r = allocate_matrix(N, N);
   a = allocate_matrix(N, N);
   b = allocate_matrix(N, N);
   psi_b = new double[N];
   psi_t = new double[N];
   psi_l = new double[N];
   psi_r = new double[N];

   for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
         F[i][j] = calc_F(i, j);
         w[i][j] = 0;
      }
   }
   for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
         a[i][j] = calc_a(i, j);
   for (int i = 0; i < N; i++)
      for (int j = 1; j < N; j++)
         b[i][j] = calc_b(i, j);
   for (int i = 1; i < N - 1; i++) {
      psi_b[i] = calc_psi(i, 0);
      psi_t[i] = calc_psi(i, N - 1);
      psi_l[i] = calc_psi(0, i);
      psi_r[i] = calc_psi(N - 1, i);
   }
   psi_b[0] = psi_l[0] = (psi_b[1] + psi_l[1]) / 2;
   psi_t[0] = psi_l[N - 1] = (psi_t[1] + psi_l[N-2]) / 2;
   psi_b[N - 1] = psi_r[0] = (psi_b[N-2] + psi_r[1]) / 2;
   psi_t[N - 1] = psi_r[N - 1] = (psi_t[N-2] + psi_r[N-2]) / 2;
   calculate_B(B);
}

int main() {
   // allocate and initialize all the input matrices
   double tau, error;
   clock_t begin = clock();
   init();
   // main cycle
   do {
      // get r
      calculate_Amat(r, w);
      matrix_sub(r, B);
      // get Ar
      calculate_Amat(Ar, r);
      // get tau
      tau = scalar_product(Ar, r) / scalar_product(Ar, Ar);
      // get tau * r
      matrix_mult_by_scalar(r, tau);
      // get new w
      matrix_sub(w, r);
      error = norm(r);
    //  printf("___error = %lf\n", error);
   } while (error > 0.000001);
   clock_t end = clock();
   double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
   printf("TIME OF EXECUTION = %lf\n", time_spent);
   return 0;
}

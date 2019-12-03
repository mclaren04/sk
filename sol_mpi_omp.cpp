#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <iostream>

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
double *l_buf, *r_buf;
double *set_scalars, *get_scalars;
int world, pid, side, anchor, p_num;

// m1 = m1 - m2
// m1 and m2 are matrices (size1 * size2)
void matrix_sub(double **m1, double **m2) {
#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++)
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++)
         m1[i][j] -= m2[i][j];
}

// multiply each element of matrix by scalar
void matrix_mult_by_scalar(double **m, double scalar) {
#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++)
#pragma omp parallel for schedule(static)
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
   if ((pid == 0 && i == 0) || (pid == p_num - 1 && i == side - 1))
      rho1 = 0.5;
   else
      rho1 = 1;
   return rho1 * rho2;
}

double scalar_product(double **u, double **v) {
   double res_aux, res = 0;
   for (int i = 0; i < side; i++) {
      res_aux = 0;
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++) {
         if (!((pid == 0 && (i == 0 && j == 0 || i == 0 && j == N-1)) ||
                  (pid == p_num-1 && (i == side-1 && j == 0 || i == side-1 && j == N-1))))
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
   double mat_next = (i == side) ? r_buf[j] : mat[i][j];
   double mat_prev = (i == 0) ? l_buf[j] : mat[i - 1][j];
   return (mat_next - mat_prev) / h1;
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
   double summand1 = (2/h1) * numeric_mlt_x(mat, side - 1, j);
   double summand2 = complex_drv_y(mat, side - 1, j);
   return summand1 + mat[side - 1][j] - summand2;
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
   return 2 * numeric_mlt_x(mat, side - 1, 0) / h1 -
      2 * numeric_mlt_y(mat, side - 1, 1) / h2 + mat[side - 1][0];
}

double corner_r_t_left(double **mat) {
   return 2 * numeric_mlt_x(mat, side - 1, N - 1) / h1 +
      2 * numeric_mlt_y(mat, side - 1, N - 1) / h2 + mat[side - 1][N - 1];
}

double corner_l_t_left(double **mat) {
   return (-2) * numeric_mlt_x(mat, 1, N - 1) / h1 +
      2 * numeric_mlt_y(mat, 0, N - 1) / h2 + mat[0][N - 1];
}

double calculate_corner_left(double **mat, int i, int j) {
   if (pid == 0 && i == 0 && j == 0) {
      return corner_l_b_left(mat);
   } else if (pid == p_num - 1 && i == side - 1 && j == 0) {
      return corner_r_b_left(mat);
   } else if (pid == p_num - 1 && i == side - 1 && j == N - 1) {
      return corner_r_t_left(mat);
   } else if (pid == 0 && i == 0 && j == N - 1) {
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

void send_recv(double **matr) {
   MPI_Request request1, request2;
   if (pid == 0) {
      MPI_Status status;
      MPI_Isend(matr[side - 1], N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request1);
      MPI_Recv(r_buf, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
      MPI_Wait(&request1, &status);
   } else if (pid == p_num - 1) {
      MPI_Status status;
      int msg1 = 2*(p_num - 1) - 1;
      int msg2 = 2*(p_num - 1) - 2;
      MPI_Isend(matr[0], N, MPI_DOUBLE, pid - 1, msg1, MPI_COMM_WORLD, &request1);
      MPI_Recv(l_buf, N, MPI_DOUBLE, pid - 1, msg2, MPI_COMM_WORLD, &status);
      MPI_Wait(&request1, &status);
   } else {
      MPI_Status status1, status2;
      int msg1 = 2*pid - 1;
      int msg2 = 2*pid;
      int msg3 = 2*pid - 2;
      int msg4 = 2*pid + 1;
      MPI_Isend(matr[0], N, MPI_DOUBLE, pid - 1, msg1, MPI_COMM_WORLD, &request1);
      MPI_Isend(matr[side - 1], N, MPI_DOUBLE, pid + 1, msg2, MPI_COMM_WORLD, &request2);
      MPI_Recv(l_buf, N, MPI_DOUBLE, pid - 1, msg3, MPI_COMM_WORLD, &status1);
      MPI_Recv(r_buf, N, MPI_DOUBLE, pid + 1, msg4, MPI_COMM_WORLD, &status2);
      MPI_Wait(&request1, &status1);
      MPI_Wait(&request2, &status2);
   }
}

// multiply A matrix by mat matrix
void calculate_Amat(double **res, double **mat) {
   send_recv(mat);
#pragma omp parallel for schedule(static)
   for (int j = 1; j < N - 1; j++) {
      if (pid == 0)
#pragma omp parallel for schedule(static)
         for (int i = 1; i < side; i++)
            res[i][j] = -delta(mat, i, j) + mat[i][j];
      else if (pid == p_num - 1)
#pragma omp parallel for schedule(static)
         for (int i = 0; i < side - 1; i++)
            res[i][j] = -delta(mat, i, j) + mat[i][j];
      else
#pragma omp parallel for schedule(static)
         for (int i = 0; i < side; i++)
            res[i][j] = -delta(mat, i, j) + mat[i][j];
   }
   for (int i = 0; i < side; i++) {
      if (!((pid == 0 && i == 0) || (pid == p_num - 1 && i == side - 1))) {
         res[i][0] = bottom_condition_left(mat, i);
         res[i][N - 1] = top_condition_left(mat, i);
      }
   }
   if (pid == 0) {
#pragma omp parallel for schedule(static)
      for (int j = 1; j < N - 1; j++)
         res[0][j] = left_condition_left(mat, j);
      res[0][0] = calculate_corner_left(mat, 0, 0);
      res[0][N - 1] = calculate_corner_left(mat, 0, N - 1);
   }
   if (pid == p_num - 1) {
#pragma omp parallel for schedule(static)
      for (int j = 1; j < N - 1; j++)
         res[side - 1][j] = right_condition_left(mat, j);
      res[side - 1][0] = calculate_corner_left(mat, side - 1, 0);
      res[side - 1][N - 1] = calculate_corner_left(mat, side - 1, N - 1);
   }
}

// calculate the right part of the numeric task
void calculate_B(double **res) {
#pragma omp parallel for schedule(static)
   for (int j = 0; j < N - 1; j++) {
      if (pid == 0)
#pragma omp parallel for schedule(static)
         for (int i = 1; i < side; i++)
            res[i][j] = F[i][j];
      else if (pid == p_num - 1)
#pragma omp parallel for schedule(static)
         for (int i = 0; i < side - 1; i++)
            res[i][j] = F[i][j];
      else
#pragma omp parallel for schedule(static)
         for (int i = 0; i < side; i++)
            res[i][j] = F[i][j];
   }
#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++) {
      res[i][0] = side_condition_right(i, 0);
      res[i][N - 1] = side_condition_right(i, N - 1);
   }
   if (pid == 0) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++)
         res[0][j] = side_condition_right(0, j);
      res[0][0] = calculate_corner_right(0, 0);
      res[0][N - 1] = calculate_corner_right(0, N - 1);
   }
   if (pid == p_num - 1) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++)
         res[side - 1][j] = side_condition_right(side - 1, j);
      res[side - 1][0] = calculate_corner_right(side - 1, 0);
      res[side - 1][N - 1] = calculate_corner_right(side - 1, N - 1);
   }
}

void init() {
   w = allocate_matrix(side, N);
   F = allocate_matrix(side, N);
   B = allocate_matrix(side, N);
   Ar = allocate_matrix(side, N);
   r = allocate_matrix(side, N);
   a = allocate_matrix(side + 1, N);
   b = allocate_matrix(side, N);
   l_buf = new double[N];
   r_buf = new double[N];
   psi_b = new double[side];
   psi_t = new double[side];
   psi_l = new double[N];
   psi_r = new double[N];
   set_scalars = new double[3];
   get_scalars = new double[3];

#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++) {
         F[i][j] = calc_F(anchor + i, j);
         w[i][j] = 0;
      }
   }
#pragma omp parallel for schedule(static)
   for (int j = 0; j < N; j++)
#pragma omp parallel for schedule(static)
      for (int i = 0; i < side + 1; i++)
         a[i][j] = calc_a(anchor + i, j);
#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++)
#pragma omp parallel for schedule(static)
      for (int j = 1; j < N; j++)
         b[i][j] = calc_b(anchor + i, j);
#pragma omp parallel for schedule(static)
   for (int i = 0; i < side; i++) {
      psi_b[i] = calc_psi(anchor + i, 0);
      psi_t[i] = calc_psi(anchor + i, N - 1);
   }
   if (pid == 0) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++)
         psi_l[j] = calc_psi(0, j);
      psi_b[0] = psi_l[0] = (psi_b[1] + psi_l[1]) / 2;
      psi_t[0] = psi_l[N - 1] = (psi_t[1] + psi_l[N-2]) / 2;
   }
   if (pid == p_num - 1) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < N; j++)
         psi_r[j] = calc_psi(N - 1, j);
      psi_b[N - 1] = psi_r[0] = (psi_b[N-2] + psi_r[1]) / 2;
      psi_t[N - 1] = psi_r[N - 1] = (psi_t[N-2] + psi_r[N-2]) / 2;
   }
   calculate_B(B);
}

int main() {
   // allocate and initialize all the input matrices
   MPI_Init(NULL, NULL);
   MPI_Comm_size(MPI_COMM_WORLD, &p_num);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   side = N / p_num;
   anchor = side * pid;
   init();
   // main cycle
   do {
      MPI_Barrier(MPI_COMM_WORLD);
      // get r
      calculate_Amat(r, w);
      matrix_sub(r, B);
      // get Ar
      calculate_Amat(Ar, r);
      // get tau
      set_scalars[0] = scalar_product(Ar, r);
      set_scalars[1] = scalar_product(Ar, Ar);
      set_scalars[2] = scalar_product(r, r);
      MPI_Reduce(set_scalars, get_scalars, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (pid == 0) {
         get_scalars[0] = get_scalars[0] / get_scalars[1];
         get_scalars[1] = sqrt(get_scalars[2]);
      }
      MPI_Bcast(get_scalars, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // get tau * r
      matrix_mult_by_scalar(r, get_scalars[0]);
      // get new w
      matrix_sub(w, r);
      if (pid == 0)
         printf("pid0: NORM = %.8lf\n", get_scalars[1]);
   } while (get_scalars[1] > 0.000001);
   MPI_Finalize();
   return 0;
}

#pragma once

namespace seq
{
/* matrix-vector multiply : y = A * x, where 
   A is symmetric and only lower half are stored */
void symMatVec(int n, double* a, double* x, double* y)
{
  int i, j;

  for (i = 0; i < n; i++) {
    double t = 0.0;
    for (j = 0; j <= i; j++)
      t += a[i * n + j] * x[j];

    for (j = i + 1; j < n; j++)
      t += a[j * n + i] * x[j];

    y[i] = t;
  }
}

/* solve Ax = b */
void solveSym(int n, double* a, double* x, double* b)
{
  /* LDLT decomposition: A = L * D * L^t */
  for (int i = 0; i < n; i++) {
    double invp = 1.0 / a[i * n + i];

    for (int j = i + 1; j < n; j++) {
      double aji = a[j * n + i];
      a[j * n + i] *= invp;

      for (int k = i + 1; k <= j; k++)
        a[j * n + k] -= aji * a[k * n + i];
    }
  }


  /* forward solve L y = b: but y is stored in x
     can be merged to the previous loop */
  for (int i = 0; i < n; i++) {
    double t = b[i];

    for (int j = 0; j < i; j++)
      t -= a[i * n + j] * x[j];

    x[i] = t;
  }

  /* backward solve D L^t x = y */
  for (int i = n - 1; i >= 0; i--) {
    double t = x[i] / a[i * n + i];

    for (int j = i + 1; j < n; j++)
      t -= a[j * n + i] * x[j];

    x[i] = t;
  }
}

}  // namespace seq
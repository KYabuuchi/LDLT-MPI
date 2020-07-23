#pragma once
#include <iostream>
#include <mpi.h>

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
void ans(int n, double* b)
{
  double* a = new double[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = b[i * n + j];
    }
  }

  /* LDLT decomposition: A = L * D * L^t */
  for (int i = 0; i < n - 1; i++) {
    double invp = 1.0 / a[i * n + i];

    for (int j = i + 1; j < n; j++) {
      double aji = a[j * n + i];
      a[j * n + i] *= invp;

      for (int k = i + 1; k <= j; k++)
        a[j * n + k] -= aji * a[k * n + i];
    }
  }
  // print
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      std::cout << a[i * n + j] << " ";
    std::cout << std::endl;
  }
  delete[] a;
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

  // print
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      std::cout << a[i * n + j] << " ";
    std::cout << std::endl;
  }

  // /* forward solve L y = b: but y is stored in x
  //    can be merged to the previous loop */
  // for (i = 0; i < n; i++) {
  //   double t = b[i];

  //   for (j = 0; j < i; j++)
  //     t -= a[i * n + j] * x[j];

  //   x[i] = t;
  // }

  // /* backward solve D L^t x = y */
  // for (i = n - 1; i >= 0; i--) {
  //   double t = x[i] / a[i * n + i];

  //   for (j = i + 1; j < n; j++)
  //     t -= a[j * n + i] * x[j];

  //   x[i] = t;
  // }
}

}  // namespace seq

namespace para
{
void solveSym(int n, double* a)
{
  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int i = myid;

  // 受信
  // ===================================================================
  if (myid != 0) {
    // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
    double data;
    MPI_Request rreq;
    MPI_Irecv((void*)&data, 1, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &rreq);
    // 待機
    MPI_Status st;
    MPI_Wait(&rreq, &st);
    // 反映
    a[i * n + i] = data;
  }

  double invp = 1.0 / a[i * n + i];
  int send_size = 0;
  int recv_size = 1;
  for (int j = i + 1; j < n; j++) {
    send_size++;
    recv_size++;

    // 受信
    // ===================================================================
    if (myid != 0) {
      // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
      double* data = new double[j];
      MPI_Request rreq;
      MPI_Irecv((void*)data, recv_size, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &rreq);
      // 待機
      MPI_Status st;
      MPI_Wait(&rreq, &st);
      // 反映
      for (int k = 0; k < recv_size; k++)
        a[j * n + (myid + k)] = data[k];
    }

    // 計算
    // ===================================================================
    double aji = a[j * n + i];
    a[j * n + i] *= invp;
    for (int k = i + 1; k <= j; k++)
      a[j * n + k] -= aji * a[k * n + i];


    // 送信
    // ===================================================================
    if (myid != nproc - 1) {
      // 送信(受信データのバッファ，データ長，データ型，送信元ID，etc)
      MPI_Request sreq;
      double* send_data = new double[send_size];
      for (int k = 0; k < send_size; k++) send_data[k] = a[j * n + (i + 1 + k)];
      MPI_Isend((void*)send_data, send_size, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, &sreq);
      // 待機
      MPI_Status st;
      MPI_Wait(&sreq, &st);
    }
  }

  // ===================================================================
  if (myid == nproc - 1) {
    // print
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++)
        std::cout << a[i * n + j] << " ";
      std::cout << std::endl;
    }
  }
}
}  // namespace para

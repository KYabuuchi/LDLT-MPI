#pragma once
#include <iostream>
#include <mpi.h>

namespace para
{
double solveSym(int n, double* a, double* b)
{
  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Cholesky Decomposition
  // ===================================================================
  {
    int i = myid;

    // 受信
    // ===================================================================
    if (myid != 0) {
      // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
      double data;
      MPI_Request rreq;
      MPI_Irecv((void*)&data, 1, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &rreq);
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
        MPI_Status st;
        MPI_Wait(&sreq, &st);
      }
    }
  }
  // // 確認
  // // ===================================================================
  // if (myid == nproc - 2) {
  //   // print
  //   for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < n; j++) {
  //       std::cout << a[i * n + j] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }


  // 交代代入1(Ly=b)
  // ===================================================================
  double y;
  {
    double tmp = b[myid];
    for (int i = 0; i < n; i++) {

      // 受信
      // ===================================================================
      if (myid > i) {
        // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
        double data = 0;
        MPI_Request rreq;
        MPI_Irecv((void*)&data, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &rreq);
        MPI_Status st;
        MPI_Wait(&rreq, &st);

        tmp -= data;
      }

      // 計算
      // ===================================================================
      if (myid == i)
        y = tmp;


      // 送信
      // ===================================================================
      if (myid < i) {
        // 送信(受信データのバッファ，データ長，データ型，送信元ID，etc)
        MPI_Request sreq;
        double data = a[n * i + myid] * y;

        MPI_Isend((void*)&data, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &sreq);
        // 待機
        MPI_Status st;
        MPI_Wait(&sreq, &st);
      }
    }
  }
  // // 確認
  // // ===================================================================
  // MPI_Barrier(MPI_COMM_WORLD);
  // std::cout << y << " @ " << myid << std::endl;

  // 交代代入2(L^t x=y)
  // ===================================================================
  double x;
  {
    double tmp = y / a[myid * n + myid];
    for (int i = n - 1; i >= 0; i--) {

      // 受信
      // ===================================================================
      if (myid < i) {
        // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
        double data = 0;
        MPI_Request rreq;
        MPI_Irecv((void*)&data, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &rreq);
        MPI_Status st;
        MPI_Wait(&rreq, &st);

        tmp -= a[i * n + myid] * data;
      }

      // 計算
      // ===================================================================
      if (myid == i)
        x = tmp;


      // 送信
      // ===================================================================
      if (myid > i) {
        // 送信(受信データのバッファ，データ長，データ型，送信元ID，etc)
        MPI_Request sreq;
        double data = x;
        MPI_Isend((void*)&data, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &sreq);
        MPI_Status st;
        MPI_Wait(&sreq, &st);
      }
    }
  }
  return x;
}
}  // namespace para

#include "mpi.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  // 一番最初に呼ぶ初期化
  MPI_Init(&argc, &argv);

  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int n = nproc + 1;
  srand(0);

  // 行列の生成
  double* a = new double[n * n];
  assert(a != NULL);

  // 下三角行列の作成
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++)
      a[i * n + j] = rand() / (RAND_MAX + 1.0);

  // 正定値対称行列を半分だけ作成
  for (int i = 0; i < n; i++) {
    double s = 0.0;
    for (int j = 0; j < i; j++) s += a[i * n + j];
    for (int j = i + 1; j < n; j++) s += a[j * n + i];
    a[i * n + i] = s + 1.0;
  }

  if (myid == 0) {

    // print
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++)
        std::cout << a[i * n + j] << " ";
      std::cout << std::endl;
    }
    std::cout << "=============" << std::endl;
    seq::ans(n, a);
    std::cout << "=============" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  para::solveSym(n, a);

  // 一番最後に呼ぶ終了宣言
  MPI_Finalize();

  return 0;

  // /* first make the solution */
  // double* xx = new double[n];
  // assert(xx != NULL);

  // for (int i = 0; i < n; i++)
  //   xx[i] = 1.0; /* or anything you like */

  // /* make right hand side b = Ax */
  // double* b = new double[n];
  // assert(b != NULL);
  // seq::symMatVec(n, a, xx, b);


  // /* solution vector, pretend to be unknown */
  // double* x = new double[n];
  // assert(x != NULL);

  // /* solve: the main computation */
  // seq::solveSym(n, a, x, b);

  // /* check error norm */
  // double e = 0;
  // for (int i = 0; i < n; i++)
  //   e += (x[i] - xx[i]) * (x[i] - xx[i]);
  // e = std::sqrt(e);

  // printf("error norm = %e\n", e);
  // printf("--- good if error is around n * 1e-16 or less\n");
}
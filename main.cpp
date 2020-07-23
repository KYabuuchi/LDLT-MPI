#include "mpi.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  // 一番最初に呼ぶ初期化
  MPI_Init(&argc, &argv);

  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int n = nproc;
  srand(0);


  // 正定値対称行列を半分だけ作成
  double* a = new double[n * n];
  assert(a != NULL);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++)
      a[i * n + j] = rand() / (RAND_MAX + 1.0);
  for (int i = 0; i < n; i++) {
    double s = 0.0;
    for (int j = 0; j < i; j++) s += a[i * n + j];
    for (int j = i + 1; j < n; j++) s += a[j * n + i];
    a[i * n + i] = s + 1.0;
  }


  // 求めるベクトルを作成
  double* xx = new double[n];
  assert(xx != NULL);
  for (int i = 0; i < n; i++) xx[i] = 1.0;


  // b=Ax を作成
  double* b = new double[n];
  assert(b != NULL);
  seq::symMatVec(n, a, xx, b);


  // 出力を入れる配列を作成
  double* x = new double[n];
  assert(x != NULL);


  if (myid == 0) {
    seq::ans(n, a, b);
  }

  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  MPI_Barrier(MPI_COMM_WORLD);
  para::solveSym(n, a, b);
  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0) {
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << " [ms]" << std::endl;
  }

  // 一番最後に呼ぶ終了宣言
  MPI_Finalize();

  return 0;

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
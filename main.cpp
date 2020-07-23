#include "para.hpp"
#include "seq.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
  // 乱数
  srand((unsigned)time(NULL));


  // 一番最初に呼ぶ初期化
  MPI_Init(&argc, &argv);


  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  // 行列のサイズはプロセス数
  int n = nproc;


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


  // 答えとなるベクトルを作成
  double* xx = new double[n];
  assert(xx != NULL);
  for (int i = 0; i < n; i++) xx[i] = 1.0;


  // b=Ax を作成
  double* b = new double[n];
  assert(b != NULL);
  seq::symMatVec(n, a, xx, b);


  // 並列計算
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
  double x = para::solveSym(n, a, b);
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();


  // 確認
  // ===================================================================
  double* output = new double[n];
  MPI_Gather((void*)&x, 1, MPI_DOUBLE, (void*)output, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (myid == 0) {
    double error = 0;
    for (int i = 0; i < n; i++)
      error += (output[i] - xx[i]) * (output[i] - xx[i]);
    error = std::sqrt(error);

    if (error > 1e-13)
      std::cout << "Xx_WARNING_xX too large error = " << error << std::endl;
    std::cout << nproc << " " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << std::endl;
  }


  // 一番最後に呼ぶ終了宣言
  MPI_Finalize();

  return 0;
}
#include <iostream>
#include <mpi.h>
#include <stdlib.h>

double cmpsum(double data)
{
  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // (log nproc)だけ繰り返す．
  for (int ix = 1; ix < nproc; ix *= 2) {
    int dst = myid ^ ix;

    // 受信(受信データのバッファ，データ長，データ型，送信元ID，etc)
    double recv;
    MPI_Request rreq;
    MPI_Irecv((void*)&recv, 1, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &rreq);

    // 送信(受信データのバッファ，データ長，データ型，送信元ID，etc)
    MPI_Request sreq;
    MPI_Isend((void*)&data, 1, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &sreq);

    // 待機
    MPI_Status st;
    MPI_Wait(&rreq, &st);
    MPI_Wait(&sreq, &st);

    // データの更新
    data += recv;
  }
  return data;
}

int main(int argc, char** argv)
{
  // 一番最初に呼ぶ初期化
  MPI_Init(&argc, &argv);

  // プロセス数と自身のIDを取得
  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  double data, sum;
  data = myid;
  sum = cmpsum(data);
  if (myid == 0)
    std::cout << sum << " " << nproc * (nproc - 1) / 2 << std::endl;


  // 一番最後に呼ぶ終了宣言
  MPI_Finalize();
}
#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>

const int N = 2;         // elementos por proceso (como en el paper: 2 ints)
const int WARMUP = 20;   // iteraciones de calentamiento
const int ITERS  = 1000; // iteraciones cronometradas
const bool CHECK = true; // validar contra MPI_Allgather del sistema

void bruck_allgather(
    MPI_Comm comm,          // MPI Communicator
    int id,                 // Process ID in Comm
    int p,                  // Number of Processes in Comm
    int* data,              // salida (n*p)
    const int* init_data,   // entrada (n)
    int n)                  // elementos por proceso
{
    
    std::memcpy(data, init_data, n * sizeof(int));

    int i = 0;
    int dist = 1;

    while (dist < p) {
        int size = n * dist;

        int sendto = id - dist;
        if (sendto >= 0) sendto = id - dist;
        else sendto = id - dist + p;

        int recvfrom = id + dist;
        if (recvfrom < p) recvfrom = id + dist;
        else recvfrom = id + dist - p;

        int max_recv = n * p - size;
        int count = size;

        if (count > max_recv) count = max_recv;

        if (count > 0) {
            MPI_Request reqs[2];
            MPI_Irecv(data + size, count, MPI_INT, recvfrom, 1000 + i, comm, &reqs[0]);
            MPI_Isend(data, count, MPI_INT, sendto, 1000 + i, comm, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }

        dist = dist << 1;
        i = i + 1;
    }

    // rotación final
    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src_block = (j + id) % p;
        std::memcpy(tmp.data() + j * n, data + src_block * n, n * sizeof(int));
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &id);

    std::vector<int> init(N), out(N * p), gold(N * p);

    for (int k = 0; k < N; ++k) init[k] = id * 100000 + k;

    if (CHECK) {
        bruck_allgather(comm, id, p, out.data(), init.data(), N);
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, comm);
        bool ok = (std::memcmp(out.data(), gold.data(), sizeof(int) * N * p) == 0);
        if (id == 0) std::cout << "Check vs MPI_Allgather: " << (ok ? "OK" : "MISMATCH") << "\n";
        MPI_Barrier(comm);
    }

    for (int w = 0; w < WARMUP; ++w) {
        bruck_allgather(comm, id, p, out.data(), init.data(), N);
        MPI_Barrier(comm);
    }

    double t0 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        bruck_allgather(comm, id, p, out.data(), init.data(), N); // comunicación + rotación
        MPI_Barrier(comm);
    }
    double t1 = MPI_Wtime();
    double time_per_iter = (t1 - t0) / (double)ITERS;

    double sum = 0.0;
    MPI_Reduce(&time_per_iter, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (id == 0) {
        double avg_over_ranks = sum / (double)p;
        std::cout << "Bruck Allgather\n";

        std::cout << "P =" << p << "  N =" << N << "  ITERS =" << ITERS << "  WARMUP =" << WARMUP << "\n";

        std::cout << "Avg time per iter (ranks): " << avg_over_ranks << " s\n";
    }

    MPI_Finalize();
    return 0;
}
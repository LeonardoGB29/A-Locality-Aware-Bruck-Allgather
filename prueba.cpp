#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

const int N = 2;         // elementos por proceso
const int WARMUP = 20;   // iteraciones de calentamiento
const int ITERS  = 1000; // iteraciones cronometradas
const bool CHECK = true; // validar resultados

// ======================================================
// BRUCK estándar
// ======================================================
void bruck_allgather(MPI_Comm comm, int id, int p, int* data, const int* init_data, int n) {
    std::memcpy(data, init_data, n * sizeof(int));

    int dist = 1;
    int step = 0;
    while (dist < p) {
        int size = n * dist;

        int sendto = (id - dist + p) % p;
        int recvfrom = (id + dist) % p;

        int max_recv = n * p - size;
        int count = std::min(size, max_recv);

        if (count > 0) {
            MPI_Request reqs[2];
            MPI_Irecv(data + size, count, MPI_INT, recvfrom, 1000 + step, comm, &reqs[0]);
            MPI_Isend(data, count, MPI_INT, sendto, 1000 + step, comm, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }

        dist <<= 1;
        step++;
    }

    // Rotación final
    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src_block = (j + id) % p;
        std::memcpy(tmp.data() + j * n, data + src_block * n, n * sizeof(int));
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));
}

// ======================================================
// Locality-Aware BRUCK (loc_bruck)
// ======================================================
void loc_bruck_allgather(
    MPI_Comm comm, int id, int p,
    MPI_Comm comm_local, int id_local, int p_local,
    int r_n, int* data, const int* init_data, int n)
{
    // 1. Fase local inicial
    std::vector<int> local_data(n * p_local);
    bruck_allgather(comm_local, id_local, p_local, local_data.data(), init_data, n);
    std::memcpy(data, local_data.data(), n * p_local * sizeof(int));

    // 2. Fase inter-región
    int steps = 0;
    for (int temp = p_local; temp < r_n; temp *= p_local) ++steps;

    for (int i = 0; i < steps; ++i) {
        int pow_next = 1;
        for (int j = 0; j <= i + 1; ++j) pow_next *= p_local;
        int size = n * pow_next;

        int dist = id_local * pow_next;

        int sendto = (id - dist + p) % p;
        int recvfrom = (id + dist) % p;

        int max_recv = n * p - size;
        int count = std::min(size, max_recv);

        if (count > 0) {
            MPI_Request reqs[2];
            MPI_Irecv(data + size, count, MPI_INT, recvfrom, 2000 + i, comm, &reqs[0]);
            MPI_Isend(data, count, MPI_INT, sendto, 2000 + i, comm, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }

        int pow_prev = 1;
        for (int j = 0; j < i; ++j) pow_prev *= p_local;
        int n_local = n * pow_prev;
        int block_offset = n * pow_next;

        if (block_offset < n * p) {
            bruck_allgather(comm_local, id_local, p_local,
                            data + block_offset, data + block_offset, n_local);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &id);

    int p_local = 2;
    int r_n = p / p_local;

    int color = id / p_local;
    MPI_Comm comm_local;
    MPI_Comm_split(comm, color, id, &comm_local);

    int id_local;
    MPI_Comm_rank(comm_local, &id_local);

    std::vector<int> init(N), out1(N * p), out2(N * p), gold(N * p);
    for (int k = 0; k < N; ++k)
        init[k] = id * 100000 + k;

    for (int w = 0; w < WARMUP; ++w) {
        bruck_allgather(comm, id, p, out1.data(), init.data(), N);
        MPI_Barrier(comm);
    }

    double t0 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        bruck_allgather(comm, id, p, out1.data(), init.data(), N);
        MPI_Barrier(comm);
    }
    double t1 = MPI_Wtime();
    double time_bruck = (t1 - t0) / ITERS;

    for (int w = 0; w < WARMUP; ++w) {
        loc_bruck_allgather(comm, id, p, comm_local, id_local, p_local, r_n, out2.data(), init.data(), N);
        MPI_Barrier(comm);
    }

    double t2 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        loc_bruck_allgather(comm, id, p, comm_local, id_local, p_local, r_n, out2.data(), init.data(), N);
        MPI_Barrier(comm);
    }
    double t3 = MPI_Wtime();
    double time_locbruck = (t3 - t2) / ITERS;

    if (CHECK) {
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, comm);
        bool ok1 = (std::memcmp(out1.data(), gold.data(), sizeof(int) * N * p) == 0);
        bool ok2 = (std::memcmp(out2.data(), gold.data(), sizeof(int) * N * p) == 0);
        if (id == 0) {
            std::cout << "Check vs MPI_Allgather:\n";
            std::cout << "  Bruck: " << (ok1 ? "OK" : "MISMATCH") << "\n";
            std::cout << "  Locality-Aware Bruck: " << (ok2 ? "OK" : "MISMATCH") << "\n";
        }
    }

    double sum1 = 0.0, sum2 = 0.0;
    MPI_Reduce(&time_bruck, &sum1, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&time_locbruck, &sum2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (id == 0) {
        double avg_bruck = sum1 / p;
        double avg_locbruck = sum2 / p;
        std::cout << "\n===== Comparación de rendimiento =====\n";
        std::cout << "P=" << p << "  P_local=" << p_local << "  Regiones=" << r_n << "\n";
        std::cout << "Bruck promedio: " << avg_bruck << " s\n";
        std::cout << "Locality-Aware Bruck promedio: " << avg_locbruck << " s\n";
        std::cout << "Aceleración: " << (avg_bruck / avg_locbruck) << "x\n";
    }

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

const int N = 2;         // elementos por proceso
const int WARMUP = 20;   // iteraciones de calentamiento
const int ITERS  = 1000; // iteraciones cronometradas
const bool CHECK = true; // validar contra MPI_Allgather


// ======================================================
// Locality-Aware BRUCK (loc_bruck)
// ======================================================
void loc_bruck_allgather(
    MPI_Comm comm, int id, int p,
    MPI_Comm comm_local, int id_local, int p_local,
    int r_n, int* data, const int* init_data, int n)
{
    // 1️ Fase local inicial
    std::vector<int> local_data(n * p_local);
    bruck_allgather(comm_local, id_local, p_local, local_data.data(), init_data, n);

    // Copiar al buffer global
    std::memcpy(data, local_data.data(), n * p_local * sizeof(int));

    // 2️ Fase inter-región
    int steps = std::ceil(std::log((double)r_n) / std::log((double)p_local));
    int offset = 0;

    for (int i = 0; i < steps; ++i) {
        int size = n * p_local * std::pow(p_local, i + 1);
        int dist = id_local * std::pow(p_local, i + 1);

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

        // 3 Re-agregación local de los datos recibidos
        int block_offset = (int)std::pow(p_local, i) * p_local * n;
        if (block_offset < n * p) {
            bruck_allgather(comm_local, id_local, p_local,
                            data + block_offset, data + block_offset, n);
        }

        offset += size;
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &id);

    // Supongamos que cada región tiene p_local procesos
    int p_local = 2;
    int r_n = p / p_local;

    // Crear comunicadores locales (por región)
    int color = id / p_local;
    MPI_Comm comm_local;
    MPI_Comm_split(comm, color, id, &comm_local);

    int id_local;
    MPI_Comm_rank(comm_local, &id_local);

    std::vector<int> init(N), out(N * p), gold(N * p);
    for (int k = 0; k < N; ++k)
        init[k] = id * 100000 + k;

    if (CHECK) {
        loc_bruck_allgather(comm, id, p, comm_local, id_local, p_local, r_n, out.data(), init.data(), N);
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, comm);
        bool ok = (std::memcmp(out.data(), gold.data(), sizeof(int) * N * p) == 0);
        if (id == 0)
            std::cout << "Check vs MPI_Allgather: " << (ok ? "OK" : "MISMATCH") << "\n";
        MPI_Barrier(comm);
    }

    for (int w = 0; w < WARMUP; ++w) {
        loc_bruck_allgather(comm, id, p, comm_local, id_local, p_local, r_n, out.data(), init.data(), N);
        MPI_Barrier(comm);
    }

    double t0 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        loc_bruck_allgather(comm, id, p, comm_local, id_local, p_local, r_n, out.data(), init.data(), N);
        MPI_Barrier(comm);
    }
    double t1 = MPI_Wtime();

    double time_per_iter = (t1 - t0) / ITERS;
    double sum = 0.0;
    MPI_Reduce(&time_per_iter, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (id == 0) {
        double avg_over_ranks = sum / p;
        std::cout << "Locality-Aware Bruck Allgather\n";
        std::cout << "P=" << p << "  P_local=" << p_local << "  Regions=" << r_n << "\n";
        std::cout << "Avg time per iter: " << avg_over_ranks << " s\n";
    }

    MPI_Finalize();
    return 0;
}

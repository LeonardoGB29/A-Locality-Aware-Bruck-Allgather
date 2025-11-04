#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>

const int N = 2;
const int ITERS = 1000;
const int WARMUP = 20;
const bool CHECK = true;

static inline int ipow(int base, int exp) {
    int r = 1;
    while (exp-- > 0) r *= base;
    return r;
}

void bruck_allgather(MPI_Comm comm, int id, int p, int* data, const int* init_data, int n) {
    std::memcpy(data, init_data, n * sizeof(int));
    const int TAG_BASE = 2200;
    int dist = 1, step = 0;
    while (dist < p) {
        int count = n * dist;
        int sendto = (id - dist + p) % p;
        int recvfrom = (id + dist) % p;
        MPI_Request reqs[2];
        MPI_Irecv(data + count, count, MPI_INT, recvfrom, TAG_BASE + step, comm, &reqs[0]);
        MPI_Isend(data, count, MPI_INT, sendto, TAG_BASE + step, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        dist <<= 1;
        ++step;
    }
    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src = (j + id) % p;
        std::memcpy(tmp.data() + j * n, data + src * n, n * sizeof(int));
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));
}

void locality_aware_bruck_allgather_corrected(MPI_Comm world, int id, int p, int* data, const int* init_data, int n) {
    MPI_Comm local_comm;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int p_l = 0, id_l = 0;
    MPI_Comm_size(local_comm, &p_l);
    MPI_Comm_rank(local_comm, &id_l);
    const int r = p / p_l;
    const int region_id = id / p_l;
    const int local_idx = id % p_l;
    if (r == 1) {
        bruck_allgather(world, id, p, data, init_data, n);
        MPI_Comm_free(&local_comm);
        return;
    }
    std::vector<int> local_data(n * p_l);
    bruck_allgather(local_comm, id_l, p_l, local_data.data(), init_data, n);
    const int TAG_BASE = 3300;
    int steps = (int)std::ceil(std::log(r) / std::log(p_l));
    std::vector<int> current_data = local_data;
    int current_count = p_l * n;
    for (int step = 0; step < steps; step++) {
        int block_size = n * ipow(p_l, step);
        int send_count = block_size * p_l;
        int region_offset = ipow(p_l, step);
        int target_region = (region_id + local_idx * region_offset) % r;
        int source_region = (region_id - local_idx * region_offset + r) % r;
        int target_rank = target_region * p_l + local_idx;
        int source_rank = source_region * p_l + local_idx;
        std::vector<int> recv_buffer(send_count);
        MPI_Sendrecv(current_data.data(), send_count, MPI_INT, target_rank, TAG_BASE + step,
                     recv_buffer.data(), send_count, MPI_INT, source_rank, TAG_BASE + step,
                     world, MPI_STATUS_IGNORE);
        std::vector<int> combined_data(current_count + send_count);
        std::memcpy(combined_data.data(), current_data.data(), current_count * sizeof(int));
        std::memcpy(combined_data.data() + current_count, recv_buffer.data(), send_count * sizeof(int));
        current_data = std::move(combined_data);
        current_count += send_count;
    }
    std::vector<int> final_data(n * p);
    bruck_allgather(local_comm, id_l, p_l, final_data.data(), current_data.data(), current_count);
    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src = (j + id) % p;
        std::memcpy(tmp.data() + j * n, final_data.data() + src * n, n * sizeof(int));
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));
    MPI_Comm_free(&local_comm);
}

double benchmark(void (*func)(MPI_Comm,int,int,int*,const int*,int),
                 MPI_Comm comm, int id, int p, int n) {
    std::vector<int> init(n), out(n * p);
    for (int k = 0; k < n; ++k) init[k] = id * 1000 + k;
    for (int i = 0; i < WARMUP; ++i)
        func(comm, id, p, out.data(), init.data(), n);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERS; ++i)
        func(comm, id, p, out.data(), init.data(), n);
    double t1 = MPI_Wtime();
    double avg = (t1 - t0) / ITERS;
    double global_avg, global_min, global_max;
    MPI_Reduce(&avg, &global_avg, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&avg, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&avg, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (id == 0) global_avg /= p;
    return (id == 0) ? global_avg : 0.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm world = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(world, &p);
    MPI_Comm_rank(world, &id);
    if (id == 0) {
        std::cout << "=== Comparativa: Bruck clásico vs Locality-Aware Bruck ===\n";
        std::cout << "Procesos: " << p << ", N=" << N << ", Iteraciones=" << ITERS << "\n";
    }
    double t_bruck = benchmark(bruck_allgather, world, id, p, N);
    double t_local = benchmark(locality_aware_bruck_allgather_corrected, world, id, p, N);
    if (id == 0) {
        std::cout << "\n=== Resultados promedio (segundos por iteración) ===\n";
        std::cout << "Bruck clásico:          " << t_bruck << " s\n";
        std::cout << "Locality-Aware Bruck:   " << t_local << " s\n";
        std::cout << "Speedup (Bruck / Local): " << (t_bruck / t_local) << "x\n";
    }
    if (CHECK) {
        std::vector<int> init(N), gold(N * p), out(N * p);
        for (int k = 0; k < N; ++k) init[k] = id * 1000 + k;
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, world);
        locality_aware_bruck_allgather_corrected(world, id, p, out.data(), init.data(), N);
        bool exact_ok = std::memcmp(out.data(), gold.data(), sizeof(int) * N * p) == 0;
        std::vector<int> gold_rotated(N * p);
        for (int j = 0; j < p; ++j) {
            int src = (j + id) % p;
            std::memcpy(gold_rotated.data() + j * N, gold.data() + src * N, N * sizeof(int));
        }
        bool rotated_ok = std::memcmp(out.data(), gold_rotated.data(), sizeof(int) * N * p) == 0;
        std::vector<int> gold_sorted = gold, out_sorted = out;
        std::sort(gold_sorted.begin(), gold_sorted.end());
        std::sort(out_sorted.begin(), out_sorted.end());
        bool content_ok = (gold_sorted == out_sorted);
        int global_exact, global_rotated, global_content;
        MPI_Allreduce(&exact_ok, &global_exact, 1, MPI_INT, MPI_LAND, world);
        MPI_Allreduce(&rotated_ok, &global_rotated, 1, MPI_INT, MPI_LAND, world);
        MPI_Allreduce(&content_ok, &global_content, 1, MPI_INT, MPI_LAND, world);
        if (id == 0) {
            std::cout << "\n=== Verificación de corrección ===\n";
            std::cout << "Exact match (same order):      " << (global_exact ? "OK" : "FAIL") << "\n";
            std::cout << "Rotated match (Bruck order):   " << (global_rotated ? "OK" : "FAIL") << "\n";
            std::cout << "Unordered content equivalence: " << (global_content ? "OK" : "FAIL") << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}


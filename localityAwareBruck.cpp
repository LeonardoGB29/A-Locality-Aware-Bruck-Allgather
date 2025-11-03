#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

const int N = 2;
const int WARMUP = 20;
const int ITERS  = 1000;
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

void locality_aware_bruck_allgather(MPI_Comm world, int id, int p, int* data, const int* init_data, int n) {
    MPI_Comm local_comm;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

    int p_l = 0, id_l = 0;
    MPI_Comm_size(local_comm, &p_l);
    MPI_Comm_rank(local_comm, &id_l);

    const int r = p / p_l;
    if (r == 1) {
        bruck_allgather(world, id, p, data, init_data, n);
        MPI_Comm_free(&local_comm);
        return;
    }

    bruck_allgather(local_comm, id_l, p_l, data, init_data, n);

    const int TAG_BASE = 3300;
    int steps = std::ceil(std::log(r) / std::log(p_l));
    
    for (int i = 0; i < steps; ++i) {
        int block_size = n * ipow(p_l, i);
        int total_size = block_size * p_l;
        
        int region_step = ipow(p_l, i);
        int target_region_offset = (id_l * region_step) % r;
        
        int local_region_id = id / p_l;
        int target_region = (local_region_id + target_region_offset) % r;
        int target_process = target_region * p_l + id_l;
        
        if (id_l > 0) {
            MPI_Request reqs[2];
            
            MPI_Isend(data, total_size, MPI_INT, target_process, 
                     TAG_BASE + i * 1000 + id, world, &reqs[0]);
            
            int source_region = (local_region_id - target_region_offset + r) % r;
            int source_process = source_region * p_l + id_l;
            MPI_Irecv(data + total_size, total_size, MPI_INT, source_process,
                     TAG_BASE + i * 1000 + source_process, world, &reqs[1]);
            
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }
        
        MPI_Barrier(local_comm);
        
        std::vector<int> local_temp(total_size * p_l);
        if (id_l > 0) {
            bruck_allgather(local_comm, id_l, p_l, local_temp.data(), 
                           data + total_size, total_size);
            
            std::memcpy(data, local_temp.data(), total_size * p_l * sizeof(int));
        } else {
            bruck_allgather(local_comm, id_l, p_l, local_temp.data(), data, total_size);
            std::memcpy(data, local_temp.data(), total_size * p_l * sizeof(int));
        }
    }

    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src = (j + id) % p;
        std::memcpy(tmp.data() + j * n, data + src * n, n * sizeof(int));
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));
    
    MPI_Comm_free(&local_comm);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm world = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(world, &p);
    MPI_Comm_rank(world, &id);

    MPI_Comm local_comm;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int p_l = 0;
    MPI_Comm_size(local_comm, &p_l);
    MPI_Comm_free(&local_comm);
    const int regions = p / p_l;

    std::vector<int> init(N), out(N * p), gold(N * p);
    for (int k = 0; k < N; ++k) init[k] = id * 100000 + k;

    if (CHECK) {
        locality_aware_bruck_allgather(world, id, p, out.data(), init.data(), N);
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, world);
        int local_ok = (std::memcmp(out.data(), gold.data(), sizeof(int) * N * p) == 0);
        int global_ok;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, world);
        if (id == 0) std::cout << "Check: " << (global_ok ? "OK" : "Fail") << std::endl;
        MPI_Barrier(world);
    }

    for (int w = 0; w < WARMUP; ++w) {
        locality_aware_bruck_allgather(world, id, p, out.data(), init.data(), N);
        MPI_Barrier(world);
    }

    double t0 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        locality_aware_bruck_allgather(world, id, p, out.data(), init.data(), N);
        MPI_Barrier(world);
    }
    double t1 = MPI_Wtime();

    double t = (t1 - t0) / ITERS;
    double sum, tmin, tmax;
    MPI_Reduce(&t, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&t, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, world);
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, world);

    if (id == 0) {
        double avg = sum / p;
        std::cout << "Locality-Aware Bruck" << std::endl;
        std::cout << "P=" << p << "  PPN=" << p_l << "  Regions=" << regions << std::endl;
        std::cout << "Avg=" << avg << "  Min=" << tmin << "  Max=" << tmax << std::endl;
    }

    MPI_Finalize();
    return 0;
}

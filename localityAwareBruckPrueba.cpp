#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

const int N = 2;
const bool CHECK = true;

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
    
    int forced_ppn = 2;
    int color = id / forced_ppn;
    MPI_Comm_split(world, color, id, &local_comm);
    
    int p_l = 0, id_l = 0;
    MPI_Comm_size(local_comm, &p_l);
    MPI_Comm_rank(local_comm, &id_l);

    const int r = p / p_l;

    if (id == 0) {
        std::cout << "DEBUG: P=" << p << ", P_l=" << p_l << ", Regions=" << r << std::endl;
    }

    if (r == 1) {
        bruck_allgather(world, id, p, data, init_data, n);
        MPI_Comm_free(&local_comm);
        return;
    }

    std::vector<int> local_data(n * p_l);
    bruck_allgather(local_comm, id_l, p_l, local_data.data(), init_data, n);

    if (id == 0) std::cout << "DEBUG: Paso 1 - Local allgather completed" << std::endl;

    const int TAG_BASE = 3300;
    
    int my_region = id / p_l;
    int target_region = (my_region + 1) % r;
    int source_region = (my_region - 1 + r) % r;
    
    int target_id = target_region * p_l + id_l;
    int source_id = source_region * p_l + id_l;

    if (id == 0) {
        std::cout << "DEBUG: Region " << my_region << " communicating with region " << target_region << std::endl;
    }

    std::vector<int> recv_data(n * p_l);
    
    MPI_Sendrecv(local_data.data(), n * p_l, MPI_INT, target_id, TAG_BASE,
                recv_data.data(), n * p_l, MPI_INT, source_id, TAG_BASE,
                world, MPI_STATUS_IGNORE);

    if (id == 0) std::cout << "DEBUG: Paso 2 - Non-local communication completed" << std::endl;

    std::vector<int> combined_data(n * p_l * 2);
    std::memcpy(combined_data.data(), local_data.data(), n * p_l * sizeof(int));
    std::memcpy(combined_data.data() + n * p_l, recv_data.data(), n * p_l * sizeof(int));

    if (id == 0) std::cout << "DEBUG: Combined data size: " << combined_data.size() << std::endl;

    std::vector<int> final_data(combined_data.size() * p_l);
    bruck_allgather(local_comm, id_l, p_l, final_data.data(), combined_data.data(), combined_data.size());

    if (id == 0) std::cout << "DEBUG: Paso 4 - Final local allgather completed" << std::endl;

    if (final_data.size() >= n * p) {
        std::memcpy(data, final_data.data(), n * p * sizeof(int));
    } else {
        std::memcpy(data, final_data.data(), final_data.size() * sizeof(int));
    }

    std::vector<int> tmp(n * p);
    for (int j = 0; j < p; ++j) {
        int src = (j + id) % p;
        if (src * n < n * p) {
            std::memcpy(tmp.data() + j * n, data + src * n, n * sizeof(int));
        }
    }
    std::memcpy(data, tmp.data(), n * p * sizeof(int));

    MPI_Comm_free(&local_comm);
    
    if (id == 0) std::cout << "DEBUG: Algorithm completed successfully" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm world = MPI_COMM_WORLD;
    int p = 0, id = 0;
    MPI_Comm_size(world, &p);
    MPI_Comm_rank(world, &id);

    std::vector<int> init(N), out(N * p), gold(N * p);
    for (int k = 0; k < N; ++k) init[k] = id * 1000 + k;

    if (CHECK) {
        if (id == 0) {
            std::cout << "=== Testing Locality-Aware Bruck ===" << std::endl;
            std::cout << "Processes: " << p << ", Data per process: " << N << " integers" << std::endl;
        }

        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, world);

        locality_aware_bruck_allgather(world, id, p, out.data(), init.data(), N);

        bool local_ok = true;
        for (int i = 0; i < N * p; i++) {
            if (out[i] != gold[i]) {
                local_ok = false;
                if (id == 0 && i < 8) {
                    std::cout << "Mismatch at position " << i << ": expected " << gold[i] << ", got " << out[i] << std::endl;
                }
                break;
            }
        }

        int global_ok;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, world);
        
        if (id == 0) {
            std::cout << "Final Check: " << (global_ok ? "PASS" : "FAIL") << std::endl;
            if (!global_ok) {
                std::cout << "Expected: ";
                for (int i = 0; i < std::min(8, N * p); i++) {
                    std::cout << gold[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Got: ";
                for (int i = 0; i < std::min(8, N * p); i++) {
                    std::cout << out[i] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

const int N = 2;
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

    std::vector<int> local_data(n * p_l);
    bruck_allgather(local_comm, id_l, p_l, local_data.data(), init_data, n);

    const int TAG_BASE = 3300;
    int steps = (int)std::ceil(std::log(r) / std::log(p_l));

    std::vector<int> current_data = local_data;
    int current_size = n * p_l;

    for (int i = 0; i < steps; ++i) {
        int block_size = n * ipow(p_l, i);
        int exchange_size = block_size * p_l;

        int region_step = ipow(p_l, i);
        int target_region_offset = (id_l * region_step) % r;

        int local_region_id = id / p_l;
        int target_region = (local_region_id + target_region_offset) % r;
        int source_region = (local_region_id - target_region_offset + r) % r;

        int target_process = target_region * p_l + id_l;
        int source_process = source_region * p_l + id_l;

        std::vector<int> recv_data(exchange_size);
        MPI_Sendrecv(current_data.data(), exchange_size, MPI_INT, target_process, TAG_BASE + i,
                     recv_data.data(), exchange_size, MPI_INT, source_process, TAG_BASE + i,
                     world, MPI_STATUS_IGNORE);

        std::vector<int> combined_data(current_size + exchange_size);
        std::memcpy(combined_data.data(), current_data.data(), current_size * sizeof(int));
        std::memcpy(combined_data.data() + current_size, recv_data.data(), exchange_size * sizeof(int));

        int new_local_size = current_size + exchange_size;
        std::vector<int> new_local_data(new_local_size * p_l);
        bruck_allgather(local_comm, id_l, p_l, new_local_data.data(), combined_data.data(), new_local_size);

        current_data = new_local_data;
        current_size = new_local_size * p_l;
    }

    std::memcpy(data, current_data.data(), n * p * sizeof(int));

    // Rotación final
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

    std::vector<int> init(N), out(N * p), gold(N * p);
    for (int k = 0; k < N; ++k) init[k] = id * 1000 + k;

    if (CHECK) {
        MPI_Allgather(init.data(), N, MPI_INT, gold.data(), N, MPI_INT, world);
        locality_aware_bruck_allgather(world, id, p, out.data(), init.data(), N);

        bool local_ok = true;
        for (int i = 0; i < N * p; ++i) {
            if (out[i] != gold[i]) { local_ok = false; break; }
        }
        int global_ok;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, world);
        if (id == 0)
            std::cout << "Final Check: " << (global_ok ? "PASS" : "FAIL") << std::endl;
    }

    MPI_Finalize();
    return 0;
}

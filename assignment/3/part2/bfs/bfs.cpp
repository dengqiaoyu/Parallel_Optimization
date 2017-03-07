#include <cstring>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include "bfs.h"

#define contains(container, element) \
  (container.find(element) != container.end())

#define CHUNK_SIZE_FRONTIER 4096

#define FRONTIER_TAG 0
#define EMPTY_TAG 1
/**
 *
 * global_frontier_sync--
 *
 * Takes a distributed graph, and a distributed frontier with each node containing
 * world_size independently produced new frontiers, and merges them such that each
 * node holds the subset of the global frontier containing local vertices.
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths) {

    // TODO 15-418/618 STUDENTS
    //
    // In this function, you should synchronize between all nodes you
    // are using for your computation. This would mean sending and
    // receiving data between nodes in a manner you see fit. Note for
    // those using async sends: you should be careful to make sure that
    // any data you send is received before you delete or modify the
    // buffers you are sending.

    int world_size = g.world_size;
    int world_rank = g.world_rank;
    // printf("process %d begin\n", world_rank);
    std::vector<int*> send_bufs;
    std::vector<int> send_idx;
    std::vector<int*> recv_bufs;

    MPI_Request* send_reqs = new MPI_Request[world_size];
    MPI_Status* probe_status = new MPI_Status[world_size];

    for (int rank = 0; rank < world_size; rank++) {
        if (rank == world_rank)
            continue;

        int remote_frontier_size =
            frontier.get_remote_frontier_size(rank);
        int *remote_frontier = frontier.get_remote_frontier(rank);
        int *remote_depths = frontier.get_remote_depths(rank);

        int* send_buf = NULL;
        send_buf = new int[remote_frontier_size * 2];
        for (int j = 0; j < remote_frontier_size; j++) {
            send_buf[2 * j] = remote_frontier[j];
            send_buf[2 * j + 1] = remote_depths[j];
        }
        send_bufs.push_back(send_buf);
        send_idx.push_back(rank);

        MPI_Isend(send_buf, remote_frontier_size * 2, MPI_INT, rank,
                  FRONTIER_TAG, MPI_COMM_WORLD, &send_reqs[rank]);
    }

    for (int rank = 0; rank < world_size; rank++) {
        if (rank == world_rank)
            continue;
        MPI_Status status;
        MPI_Probe(rank, FRONTIER_TAG, MPI_COMM_WORLD, &probe_status[rank]);
        int num_vals = 0;
        MPI_Get_count(&probe_status[rank], MPI_INT, &num_vals);

        int* recv_buf = new int[num_vals];
        recv_bufs.push_back(recv_buf);

        MPI_Recv(recv_buf, num_vals, MPI_INT, probe_status[rank].MPI_SOURCE,
                 probe_status[rank].MPI_TAG, MPI_COMM_WORLD, &status);

        for (int j = 0; j < num_vals; j += 2) {
            int v = recv_buf[j];
            int depth = recv_buf[j + 1];
            assert(g.get_vertex_owner_rank(v) == world_rank);
            int depths_idx = v - g.start_vertex;
            if (depths[depths_idx] > depth) {
                // printf("depths[%d]: %d, depth: %d\n",
                //        depths_idx, depths[depths_idx], depth);
                depths[depths_idx] = depth;
                Vertex *local_frontier = frontier.get_local_frontier();
                int *local_depths = frontier.get_local_depths();
                int local_depths_size = frontier.get_local_frontier_size();
                int local_depths_idx = 0;
                for (local_depths_idx = 0;
                        local_depths_idx < local_depths_size;
                        local_depths_idx++) {
                    if (local_frontier[local_depths_idx] == v)
                        break;
                }
                assert(local_depths_idx < local_depths_size);
                local_depths[local_depths_idx] = depth;
            } else if (depths[depths_idx] == NOT_VISITED_MARKER) {
                depths[depths_idx] = depth;
                frontier.add(world_rank, v, depth);
            }
        }
    }

    for (size_t i = 0; i < send_bufs.size(); i++) {
        MPI_Status status;
        MPI_Wait(&send_reqs[send_idx[i]], &status);
        delete(send_bufs[i]);
    }

    for (size_t i = 0; i < recv_bufs.size(); i++) {
        delete(recv_bufs[i]);
    }

    delete(send_reqs);
    delete(probe_status);

    // printf("process %d end\n", world_rank);
}

/*
 * bfs_step --
 *
 * Carry out one step of a distributed bfs
 *
 * depths: current state of depths array for local vertices
 * current_frontier/next_frontier: copies of the distributed frontier structure
 *
 * NOTE TO STUDENTS: We gave you this function as a stub.  Feel free
 * to change as you please (including the arguments)
 */
void bfs_step(DistGraph &g, int *depths,
              DistFrontier &current_frontier,
              DistFrontier &next_frontier) {
    int start_vertex = (int) g.start_vertex;
    int end_vertex = (int) g.end_vertex;

    int frontier_size = current_frontier.get_local_frontier_size();
    Vertex* local_frontier = current_frontier.get_local_frontier();
    int out_dst_num = g.out_edges.size();
    #pragma omp parallel
    {
        int *local_thread_frontier = (int *)malloc(out_dst_num * sizeof(int));
        int *local_thread_frontier_depths =
            (int *)malloc(out_dst_num * sizeof(int));
        int local_thread_index = 0;

        #pragma omp for schedule(dynamic, CHUNK_SIZE_FRONTIER)
        for (int i = 0; i < frontier_size; i++) {
            int node = local_frontier[i];
            int node_local_idx = node - start_vertex;
            int outgoing_edge_num = g.out_edge_dst_size[node_local_idx];
            for (int edge_idx = 0; edge_idx < outgoing_edge_num; edge_idx++) {
                int outgoing = g.out_edge_dst[node_local_idx][edge_idx];
                if (outgoing >= start_vertex && outgoing <= end_vertex) {
                    int outgoing_local_idx = outgoing - start_vertex;
                    if (depths[outgoing_local_idx] == NOT_VISITED_MARKER) {
                        depths[outgoing_local_idx] = depths[node_local_idx] + 1;
                        local_thread_frontier_depths[local_thread_index] =
                            depths[outgoing_local_idx];
                        local_thread_frontier[local_thread_index++] = outgoing;
                    }
                } else {
                    int rank = g.get_vertex_owner_rank(outgoing);
                    next_frontier.add(rank, outgoing,
                                      depths[node_local_idx] + 1);
                }
            }
        }

        int old_index, new_index, is_break;
        do {
            int *old_index_ptr = next_frontier.get_local_frontier_size_ptr();
            old_index = *old_index_ptr;
            new_index = old_index + local_thread_index;
            is_break =
                __sync_bool_compare_and_swap(old_index_ptr,
                                             old_index, new_index);
        } while (!is_break);
        memcpy(next_frontier.get_local_frontier() + old_index,
               local_thread_frontier,
               local_thread_index * sizeof(int));
        memcpy(next_frontier.get_local_depths() + old_index,
               local_thread_frontier_depths,
               local_thread_index * sizeof(int));
        free(local_thread_frontier);
        free(local_thread_frontier_depths);
    }
}

/*
 * bfs --
 *
 * Execute a distributed BFS on the distributed graph g
 *
 * Upon return, depths[i] should be the distance of the i'th local
 * vertex from the BFS root node
 */
void bfs(DistGraph &g, int *depths) {
    DistFrontier current_frontier(g.vertices_per_process, g.world_size,
                                  g.world_rank);
    DistFrontier next_frontier(g.vertices_per_process, g.world_size,
                               g.world_rank);

    DistFrontier *cur_front = &current_frontier,
                  *next_front = &next_frontier;

    // Initialize all the depths to NOT_VISITED_MARKER.
    // Note: Only storing local vertex depths.
    for (int i = 0; i < g.vertices_per_process; ++i )
        depths[i] = NOT_VISITED_MARKER;

    // Add the root node to the frontier
    int offset = g.start_vertex;
    if (g.get_vertex_owner_rank(ROOT_NODE_ID) == g.world_rank) {
        current_frontier.add(g.get_vertex_owner_rank(ROOT_NODE_ID), ROOT_NODE_ID, 0);
        depths[ROOT_NODE_ID - offset] = 0;
    }

    int iterarion = 0;
    while (true) {
        iterarion++;
        bfs_step(g, depths, *cur_front, *next_front);

        // if (iterarion == 5) {
        //     printf("break in here\n");
        //     break;
        // }
        // // this is a global empty check, not a local frontier empty check.
        // // You will need to implement is_empty() in ../dist_graph.h
        if (next_front->is_empty(iterarion)) {
            printf("!!!This one is never executed world_rank: %d!!!\n", g.world_rank);
            break;
        }

        printf("g.world_rank: %d, line 238\n", g.world_rank);
        // exchange frontier information
        global_frontier_sync(g, *next_front, depths);

        printf("g.world_rank: %d, line 242\n", g.world_rank);
        DistFrontier *temp = cur_front;
        cur_front = next_front;
        next_front = temp;
        next_front -> clear();

        printf("@@@@@@@@@@@@@@@@@\n");
        for (int i = 0; i < g.vertices_per_process; ++i ) {
            printf("process %d iterateion: %d   0->%d:%d\n",
                   g.world_rank, iterarion, i + g.start_vertex, depths[i]);
        }
        printf("@@@@@@@@@@@@@@@@@\n");
        int ii = 0;
        while (ii < 100) {
            ii++;
        }
    }
    for (int i = 0; i < g.vertices_per_process; ++i ) {
        printf("process %d    0->%d:%d\n",
               g.world_rank, i + g.start_vertex, depths[i]);
    }
    printf("process %d exit normally\n", g.world_rank);
}


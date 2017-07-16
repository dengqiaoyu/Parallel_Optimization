/**
 * @file bfs.cpp
 * @brief this file contains BFS algorithm
 */
#include <cstring>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include <unistd.h>
#include "bfs.h"

#define contains(container, element) \
  (container.find(element) != container.end())

#define CHUNK_SIZE_FRONTIER 4096

#define FRONTIER_TAG 0 /* message tag for adding frontier */
#define EMPTY_TAG 1 /* message tag for empty confirm */

int check_validation(DistGraph &g, DistFrontier &next_front, int iteration);

/**
 * Takes a distributed graph, and a distributed frontier with each node
 * containing world_size independently produced new frontiers, and merges them
 * such that each node holds the subset of the global frontier containing local
 * vertices.
 * @param g         graph type
 * @param frontier  local frontier
 * @param depths    the distance to root node
 * @param iteration the current iteration
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths,
                          int iteration) {

    int world_size = g.world_size;
    int world_rank = g.world_rank;
    std::vector<int*> send_bufs;
    std::vector<int> send_idx;
    std::vector<int*> recv_bufs;

    MPI_Request* send_reqs = new MPI_Request[world_size];
    /* to get the size of array that will be received */
    MPI_Status* probe_status = new MPI_Status[world_size];

    /* send every machine their new frontier information */
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
        /* push into vector for future deleting */
        send_bufs.push_back(send_buf);
        send_idx.push_back(rank);

        /* non blocking, needs MPI_wait to sync*/
        MPI_Isend(send_buf, remote_frontier_size * 2, MPI_INT, rank,
                  FRONTIER_TAG, MPI_COMM_WORLD, &send_reqs[rank]);
    }

    /* receive new frontier from other machines */
    for (int rank = 0; rank < world_size; rank++) {
        if (rank == world_rank)
            continue;
        MPI_Status status;
        /* get array size that will be received */
        MPI_Probe(rank, FRONTIER_TAG, MPI_COMM_WORLD, &probe_status[rank]);
        int num_vals = 0;
        MPI_Get_count(&probe_status[rank], MPI_INT, &num_vals);

        /* for future delete */
        int* recv_buf = new int[num_vals];
        recv_bufs.push_back(recv_buf);

        MPI_Recv(recv_buf, num_vals, MPI_INT, probe_status[rank].MPI_SOURCE,
                 probe_status[rank].MPI_TAG, MPI_COMM_WORLD, &status);

        /* add new node into local frontier */
        for (int j = 0; j < num_vals; j += 2) {
            int v = recv_buf[j];
            int depth = recv_buf[j + 1];
            if (g.get_vertex_owner_rank(v) != world_rank)
                assert(g.get_vertex_owner_rank(v) == world_rank);
            int depths_idx = v - g.start_vertex;
            /* If node has not been added into frontier */
            if (depths[depths_idx] == NOT_VISITED_MARKER) {
                depths[depths_idx] = depth;
                frontier.add(world_rank, v, depth);
            }
        }
    }

    /* wait every machine to receive their message */
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
}

/**
 * Carry out one step of a distributed bfs
 * @param g                   graph type
 * @param depths              current state of depths array for local vertices
 * @param current_frontier    current copy of the distributed frontier
 *                            structure
 * @param next_frontier       new copy of the distributed frontier structure
 * @param already_in_frontier array that indicate whether a node is already
 *                            added into frontier
 * @param iteration           the current iteration
 */
void bfs_step(DistGraph &g, int *depths,
              DistFrontier &current_frontier,
              DistFrontier &next_frontier, int *already_in_frontier,
              int iteration) {
    int start_vertex = (int) g.start_vertex;
    int frontier_size = current_frontier.get_local_frontier_size();
    Vertex* local_frontier = current_frontier.get_local_frontier();
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, CHUNK_SIZE_FRONTIER)
        for (int i = 0; i < frontier_size; i++) {
            int node = local_frontier[i];
            int node_local_idx = node - start_vertex;
            int outgoing_edge_num = g.out_edge_dst_size[node_local_idx];
            for (int edge_idx = 0; edge_idx < outgoing_edge_num; edge_idx++) {
                int outgoing = g.out_edge_dst[node_local_idx][edge_idx];
                /* if node is in the local range */
                if (g.get_vertex_owner_rank(outgoing) == g.world_rank) {
                    int outgoing_local_idx = outgoing - start_vertex;
                    if (depths[outgoing_local_idx] == NOT_VISITED_MARKER) {
                        #pragma omp critical
                        {
                            depths[outgoing_local_idx] =
                            depths[node_local_idx] + 1;
                            next_frontier.add(g.world_rank, outgoing,
                            depths[outgoing_local_idx]);
                        }
                    }
                } else {
                    /* if node is in other machine */
                    int rank = g.get_vertex_owner_rank(outgoing);
                    #pragma omp critical
                    {
                        /**
                         * fill already_in_frontier to avoid adding this node
                         * again in the future
                         */
                        if (already_in_frontier[outgoing] == 0) {
                            next_frontier.add(rank, outgoing,
                                              depths[node_local_idx] + 1);
                            already_in_frontier[outgoing] = 1;
                        }
                    }
                }
            }
        }
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
/**
 * Execute a distributed BFS on the distributed graph g
 * @param g      graph type
 * @param depths current state of depths array for local vertices
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
        current_frontier.add(g.get_vertex_owner_rank(ROOT_NODE_ID),
                             ROOT_NODE_ID, 0);
        depths[ROOT_NODE_ID - offset] = 0;
    }

    int iterarion = 0;
    int *already_in_frontier = (int *)malloc(sizeof(int) * g.total_vertices());
    memset(already_in_frontier, 0, sizeof(int) * g.total_vertices());
    while (true) {
        iterarion++;
        bfs_step(g, depths, *cur_front, *next_front, already_in_frontier,
                 iterarion);

        if (next_front->is_empty(iterarion)) {
            break;
        }

        // exchange frontier information
        global_frontier_sync(g, *next_front, depths, iterarion);

        DistFrontier *temp = cur_front;
        cur_front = next_front;
        next_front = temp;
        next_front -> clear();
    }
}

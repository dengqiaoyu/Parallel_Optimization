/**
 * @file bfs.cpp
 * @brief This file contains three different implementation for bfs algorithm.
 * @author Changkai Zhou(zchangka) Qiaoyu Deng(qdeng)
 * @bug No known bugs
 */

#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
/* when initialize nodes, the chunk size of openMP */
#define CHUNK_SIZE_INIT 4096
/* the chunk size for OpenMP when use top down algorithm */
#define CHUNK_SIZE_FRONTIER 32
/* the chunk size for OpenMP when use bottom up algorithm */
#define CHUNK_SIZE_UNVISITED 1024
/* the threshold of number of frontier for hybrid algorithm */
#define THREASHOLD 1000000
// #define VERBOSE /* when defined, print every iteration's iteration */

/**
 * Clear the entire vertex_set.
 * @param list the pointer to the vertex list
 */
void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

/**
 * Initialize the vertex list.
 * @param list  the pointer to the vertex list
 * @param count the count of vertex that needs to be stored in list
 */
void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

/**
 * One step for top down algorithm, search frontier node's adjacent nodes and
 * add them into new frontier.
 * @param g            graph type
 * @param frontier     the frontier set for current iteration
 * @param new_frontier the frontier set for next iteration
 * @param distances    the distance to root that needs to be calculated
 */
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances) {
    int num_nodes = g->num_nodes;
    /* create a bunch of threads to use in the future */
    #pragma omp parallel
    {
        int frontier_count = frontier->count;
        /* to avoid high contention, use local array for temp use*/
        int *local_frontier = (int *)malloc(num_nodes * sizeof(int));
        int local_index = 0;

        /**
         * let threads run for loop dynamically, and each of them will choose a
         * chunk of loop to run in parallel
         */
        #pragma omp for schedule(dynamic, CHUNK_SIZE_FRONTIER)
        for (int i = 0; i < frontier_count; i++) {
            /* iterate all the frontier nodes' adjacent nodes */
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                /* put those nodes into new frontier set */
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = distances[node] + 1;
                    int index = local_index++;
                    local_frontier[index] = outgoing;
                }
            }
        }

        /* write local result back to the global variable */
        int old_index, new_index, is_break;
        do {
            old_index = new_frontier->count;
            new_index = old_index + local_index;
            is_break =
                __sync_bool_compare_and_swap(&new_frontier->count,
                                             old_index, new_index);
        } while (!is_break);
        memcpy(new_frontier->vertices + old_index,
               local_frontier,
               local_index * sizeof(int));
        free(local_frontier);
    }
}

/**
 * BFS algorithm from root to the end for finding shortest path to root node.
 * This algorithm will first add root node into frontier and then expand
 * frontier from its adjacent node adding them into frontier too, this process
 * will continue until there is no more new node being added into frontier.
 * @param graph graph type
 * @param sol   the distance to root for every node
 */
void bfs_top_down(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(static, CHUNK_SIZE_INIT)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n",
               frontier->count, end_time - start_time);
#endif
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

/**
 * One step for bottom up algorithm, every node check whether is is connected
 * with a frontier, if it is true, then it will be added into new frontier set.
 * @param g              graph type
 * @param distances      the distance array for current iteration
 * @param new_distances  the distance array for next iteration
 * @param frontier_count the number of frontier nodes for last iteration
 */
void bfs_bottom_up_step(Graph g, int* distances, int* new_distances,
                        int *frontier_count) {
    int num_nodes = g->num_nodes;
    /* create a bunch of threads to use in the future */
    #pragma omp parallel
    {
        int local_frontier_count = 0;
        /* use no wait to let thread run faster */
        #pragma omp for schedule(dynamic, CHUNK_SIZE_UNVISITED) nowait
        for (int i = 1; i < num_nodes; i++) {
            /* iterate all the nodes inside the graph */
            if (incoming_size(g, i) == 0) {
                continue;
            } else if (distances[i] != NOT_VISITED_MARKER) {
                /* update the last iteration result to new distance */
                if (new_distances[i] == NOT_VISITED_MARKER)
                    new_distances[i] = distances[i];
                continue;
            }

            int start_edge = g->incoming_starts[i];
            int end_edge = (i == num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];
            /* iterate all of neighbors */
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                int dist_incoming = distances[incoming];
                /* do not add itself to the frontier */
                if (dist_incoming != NOT_VISITED_MARKER && i != incoming) {
                    new_distances[i] = dist_incoming + 1;
                    local_frontier_count++;
                    break;
                }
            }
        }

        /* write local result to global result */
        int old_index, new_index, is_break;
        do {
            old_index = *frontier_count;
            new_index = old_index + local_frontier_count;
            is_break =
                __sync_bool_compare_and_swap(frontier_count,
                                             old_index, new_index);
        } while (!is_break);
    }
}

/**
 * BFS algorithm from each node inside a graph to their own adjacent nodes to
 * determine whether they should be added into the frontier set, if one of their
 * adjacent is in the frontier, then they should be added into frontier set.
 * @param graph graph type
 * @param sol   the distance to root for every node
 */
void bfs_bottom_up(Graph graph, solution* sol) {

    int frontier_count = 0;
    int num_nodes = graph->num_nodes;

    int *distances = (int *)malloc(num_nodes * sizeof(int));
    memset(distances, 0xff, num_nodes * sizeof(int));
    int *new_distances = (int *)malloc(num_nodes * sizeof(int));
    memset(new_distances, 0xff, num_nodes * sizeof(int));

    /**
     * Add root's neighbors to the frontier, if the value in distance array is
     * not -1, it means this is a frontier node.
     */
    int root_start_edge = graph->outgoing_starts[ROOT_NODE_ID];
    int root_end_edge = graph->outgoing_starts[ROOT_NODE_ID + 1];
    for (int neighbor = root_start_edge; neighbor < root_end_edge; neighbor++) {
        int outgoing = graph->outgoing_edges[neighbor];
        distances[outgoing] = 1;
        frontier_count++;
    }
    distances[ROOT_NODE_ID] = 0;
    new_distances[ROOT_NODE_ID] = 0;

    while (frontier_count != 0) {
        frontier_count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        bfs_bottom_up_step(graph, distances, new_distances, &frontier_count);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier =%-10d %.4f sec\n",
               frontier_count, end_time - start_time);
#endif
        int *swap_tmp = NULL;
        swap_tmp = distances;
        distances = new_distances;
        new_distances = swap_tmp;
    }
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    /* copy result to the result pointer array */
    memcpy(sol->distances, new_distances, sizeof(int) * num_nodes);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("duration: %.4f\n", end_time - start_time);
#endif
}

/**
 * BFS algorithm for hybrid method, in this algorithm, both top down and bottom
 * up method are used accdrding to the size of frontier.
 * @param g             graph type
 * @param frontier      the frontier array for current iteration
 * @param new_frontier  the frontier array for next iteration
 * @param distances     the distance array for current iteration
 * @param new_distances the distance array for next iteration
 */
void bfs_hybrid_step(Graph g,
                     vertex_set *frontier, vertex_set *new_frontier,
                     int *distances, int *new_distances) {
    int frontier_count = frontier->count;
    /**
     * THREASHOLD is 1000000, when frontier is very small, use top down; when it
     * is very big, use bottom up
     */
    if (frontier_count > THREASHOLD) {
        int num_nodes = g->num_nodes;
        #pragma omp parallel
        {
            int local_frontier_count = 0;
            int *local_frontier = (int *)malloc(num_nodes * sizeof(int));
            #pragma omp for schedule(dynamic, CHUNK_SIZE_UNVISITED) nowait
            for (int i = 1; i < num_nodes; i++) {
                if (incoming_size(g, i) == 0) {
                    continue;
                } else if (distances[i] != NOT_VISITED_MARKER) {
                    if (new_distances[i] == NOT_VISITED_MARKER)
                        new_distances[i] = distances[i];
                    continue;
                }

                int start_edge = g->incoming_starts[i];
                int end_edge = (i == num_nodes - 1)
                               ? g->num_edges
                               : g->incoming_starts[i + 1];

                for (int neighbor = start_edge;
                        neighbor < end_edge;
                        neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    int dist_incoming = distances[incoming];
                    if (dist_incoming != NOT_VISITED_MARKER && i != incoming) {
                        new_distances[i] = dist_incoming + 1;
                        local_frontier[local_frontier_count++] = i;
                        break;
                    }
                }
            }

            int old_index, new_index, is_break;
            do {
                old_index = new_frontier->count;
                new_index = old_index + local_frontier_count;
                is_break =
                    __sync_bool_compare_and_swap(&new_frontier->count,
                                                 old_index, new_index);
            } while (!is_break);
            memcpy(new_frontier->vertices + old_index,
                   local_frontier,
                   local_frontier_count * sizeof(int));
            free(local_frontier);
        }
        return;
    } else {
        int num_nodes = g->num_nodes;
        #pragma omp parallel
        {
            int frontier_count = frontier->count;
            int *local_frontier = (int *)malloc(num_nodes * sizeof(int));
            int local_index = 0;

            #pragma omp for schedule(dynamic, CHUNK_SIZE_FRONTIER)
            for (int i = 0; i < frontier_count; i++) {
                int node = frontier->vertices[i];
                if (new_distances[node] == NOT_VISITED_MARKER)
                    new_distances[node] = distances[node];
                int start_edge = g->outgoing_starts[node];
                int end_edge = (node == num_nodes - 1)
                               ? g->num_edges
                               : g->outgoing_starts[node + 1];

                for (int neighbor = start_edge;
                        neighbor < end_edge;
                        neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];
                    if (distances[outgoing] == NOT_VISITED_MARKER) {
                        distances[outgoing] = distances[node] + 1;
                        new_distances[outgoing] = distances[outgoing];
                        int index = local_index++;
                        local_frontier[index] = outgoing;
                    }
                }
            }

            int old_index, new_index, is_break;
            do {
                old_index = new_frontier->count;
                new_index = old_index + local_index;
                is_break =
                    __sync_bool_compare_and_swap(&new_frontier->count,
                                                 old_index, new_index);
            } while (!is_break);
            memcpy(new_frontier->vertices + old_index,
                   local_frontier,
                   local_index * sizeof(int));
            free(local_frontier);
        }
        return;
    }
}

/**
 * BFS algorithm for hybrid method, in this algorithm, both top down and bottom
 * up method are used accdrding to the size of frontier.
 * @param graph graph type
 * @param sol   the distance to root for every node
 */
void bfs_hybrid(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    int num_nodes = graph->num_nodes;
    int *distances = (int *)malloc(num_nodes * sizeof(int));
    memset(distances, 0xff, num_nodes * sizeof(int));
    int *new_distances = (int *)malloc(num_nodes * sizeof(int));
    memset(new_distances, 0xff, num_nodes * sizeof(int));

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    distances[ROOT_NODE_ID] = 0;
    new_distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        bfs_hybrid_step(graph,
                        frontier, new_frontier,
                        distances, new_distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n",
               frontier->count, end_time - start_time);
#endif
        int *swap_tmp1 = distances;
        distances = new_distances;
        new_distances = swap_tmp1;

        vertex_set* swap_tmp2 = frontier;
        frontier = new_frontier;
        new_frontier = swap_tmp2;
    }
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    memcpy(sol->distances, new_distances, sizeof(int) * num_nodes);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("duration: %.4f\n", end_time - start_time);
#endif
}

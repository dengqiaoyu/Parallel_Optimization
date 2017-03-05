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
#define CHUNK_SIZE_INIT 4096
#define CHUNK_SIZE_FRONTIER 32
#define CHUNK_SIZE_UNVISITED 4096
#define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// // Take one step of "top-down" BFS.  For each vertex on the frontier,
// // follow all outgoing edges, and add all neighboring vertices to the
// // new_frontier.
// void top_down_step(
//     Graph g,
//     vertex_set* frontier,
//     vertex_set* new_frontier,
//     int* distances) {

//     for (int i = 0; i < frontier->count; i++) {

//         int node = frontier->vertices[i];

//         int start_edge = g->outgoing_starts[node];
//         int end_edge = (node == g->num_nodes - 1)
//                        ? g->num_edges
//                        : g->outgoing_starts[node + 1];

//         // attempt to add all neighbors to the new frontier
//         for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
//             int outgoing = g->outgoing_edges[neighbor];

//             if (distances[outgoing] == NOT_VISITED_MARKER) {
//                 distances[outgoing] = distances[node] + 1;
//                 int index = new_frontier->count++;
//                 new_frontier->vertices[index] = outgoing;
//             }
//         }
//     }
// }

// // Implements top-down BFS.
// //
// // Result of execution is that, for each node in the graph, the
// // distance to the root is stored in sol.distances.
// void bfs_top_down(Graph graph, solution* sol) {

//     vertex_set list1;
//     vertex_set list2;
//     vertex_set_init(&list1, graph->num_nodes);
//     vertex_set_init(&list2, graph->num_nodes);

//     vertex_set* frontier = &list1;
//     vertex_set* new_frontier = &list2;

//     // initialize all nodes to NOT_VISITED
//     for (int i = 0; i < graph->num_nodes; i++)
//         sol->distances[i] = NOT_VISITED_MARKER;

//     // setup frontier with the root node
//     frontier->vertices[frontier->count++] = ROOT_NODE_ID;
//     sol->distances[ROOT_NODE_ID] = 0;

//     while (frontier->count != 0) {

// #ifdef VERBOSE
//         double start_time = CycleTimer::currentSeconds();
// #endif

//         vertex_set_clear(new_frontier);

//         top_down_step(graph, frontier, new_frontier, sol->distances);

// #ifdef VERBOSE
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif

//         // swap pointers
//         vertex_set* tmp = frontier;
//         frontier = new_frontier;
//         new_frontier = tmp;
//     }
// }

// void top_down_step(
//     Graph g,
//     vertex_set* frontier,
//     vertex_set* new_frontier,
//     int* distances) {
//     int num_nodes = g->num_nodes;
//     #pragma omp parallel
//     {
//         int frontier_count = frontier->count;
//         int *local_frontier = (int *)malloc(num_nodes * sizeof(int));
//         int local_index = 0;
//         int id, i, Nthrds, istart, iend;
//         id = omp_get_thread_num();
//         Nthrds = omp_get_num_threads();
//         istart = id * frontier_count / Nthrds;
//         iend = (id + 1) * frontier_count / Nthrds;
//         if (id == Nthrds - 1) iend = frontier_count;
//         for (int i = istart; i < iend; i++) {
//             int node = frontier->vertices[i];

//             int start_edge = g->outgoing_starts[node];
//             int end_edge = (node == num_nodes - 1)
//                            ? num_nodes
//                            : g->outgoing_starts[node + 1];
//             for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
//                 int outgoing = g->outgoing_edges[neighbor];
//                 if (distances[outgoing] == NOT_VISITED_MARKER) {
//                     distances[outgoing] = distances[node] + 1;
//                     local_frontier[local_index++] = outgoing;
//                 }
//             }
//         }
//         int old_index, new_index, is_break;
//         do {
//             old_index = new_frontier->count;
//             new_index = old_index + local_index;
//             is_break =
//                 __sync_bool_compare_and_swap(&new_frontier->count,
//                                              old_index, new_index);
//         } while (!is_break);
//         memcpy(new_frontier->vertices + old_index, local_frontier, local_index);
//     }
// }

void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances) {
    int num_nodes = g->num_nodes;
    #pragma omp parallel
    {
        int frontier_count = frontier->count;
        int *local_frontier = (int *)malloc(num_nodes * sizeof(int));
        int local_index = 0;

        #pragma omp for schedule(dynamic, CHUNK_SIZE_FRONTIER)
        for (int i = 0; i < frontier_count; i++) {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = distances[node] + 1;
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
}

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
        // printf("count: %d\n", frontier->count);
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up_step(Graph g, int* distances,
                        vertex_set* node_status, vertex_set* new_node_status,
                        vertex_set* unvisited_set, vertex_set* new_unvisited_set,
                        int *frontier_count) {
    int unvisited_set_count = unvisited_set->count;
    int num_nodes = g->num_nodes;
    #pragma omp parallel
    {
        int* local_unvisited_set =
            (int*) malloc(unvisited_set_count * sizeof(int));
        int local_unvisited_set_count = 0;
        int local_frontier_count = 0;
        #pragma omp for schedule(dynamic, CHUNK_SIZE_UNVISITED)
        for (int i = 0; i < unvisited_set_count; i++) {
            int is_frontier = 0;
            int node = unvisited_set->vertices[i];
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                if (node_status->vertices[incoming] == 1) {
                    distances[node] = distances[incoming] + 1;
                    local_frontier_count++;
                    new_node_status->vertices[node] = 1;
                    is_frontier = 1;
                    break;
                }
            }

            if (is_frontier == 0) {
                local_unvisited_set[local_unvisited_set_count++] = node;
            }
        }

        int old_index, new_index, is_break;
        do {
            old_index = new_unvisited_set->count;
            new_index = old_index + local_unvisited_set_count;
            is_break =
                __sync_bool_compare_and_swap(&new_unvisited_set->count,
                                             old_index, new_index);
        } while (!is_break);
        memcpy(new_unvisited_set->vertices + old_index,
               local_unvisited_set,
               local_unvisited_set_count * sizeof(int));
        do {
            old_index = *frontier_count;
            new_index = old_index + local_frontier_count;
            is_break =
                __sync_bool_compare_and_swap(frontier_count,
                                             old_index, new_index);
        } while (!is_break);
        free(local_unvisited_set);
    }
}

void bfs_bottom_up(Graph graph, solution* sol) {
    // 15-418/618 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    int frontier_count = 0;
    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set list4;
    int num_nodes = graph->num_nodes;
    vertex_set_init(&list1, num_nodes);
    vertex_set_init(&list2, num_nodes);
    vertex_set_init(&list3, num_nodes);
    vertex_set_init(&list4, num_nodes);

    vertex_set* node_status = &list1;
    vertex_set* new_node_status = &list2;
    vertex_set* unvisited_set = &list3;
    vertex_set* new_unvisited_set = &list4;
    memset(node_status->vertices, 0, sizeof(int) * num_nodes);
    memset(new_node_status->vertices, 0, sizeof(int) * num_nodes);
    memset(unvisited_set->vertices, 0, sizeof(int) * num_nodes);
    memset(new_unvisited_set->vertices, 0, sizeof(int) * num_nodes);

    // #pragma omp parallel for schedule(static, CHUNK_SIZE_INIT)
    int non_zero_outgoing_node_count = 0;
    for (int i = 1; i < num_nodes; i++) {
        if (incoming_size(graph, i))
            unvisited_set->vertices[non_zero_outgoing_node_count++] = i;
    }
    unvisited_set->count = non_zero_outgoing_node_count;
    // printf("non_zero_outgoing_node_count: %d\n", non_zero_outgoing_node_count);
    // exit(1);

    node_status->vertices[frontier_count++] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier_count != 0) {
        frontier_count = 0;
        new_unvisited_set->count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        bfs_bottom_up_step(graph, sol->distances,
                           node_status, new_node_status,
                           unvisited_set, new_unvisited_set,
                           &frontier_count);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("unvisited set=%-10d %.4f sec\n",
               unvisited_set->count, end_time - start_time);
#endif
        // printf("frontier_count: %d\n", frontier_count);
        vertex_set* tmp = node_status;
        node_status = new_node_status;
        new_node_status = tmp;

        tmp = unvisited_set;
        unvisited_set = new_unvisited_set;
        new_unvisited_set = tmp;
    }
}

void bfs_hybrid(Graph graph, solution* sol) {
    // 15-418/618 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}

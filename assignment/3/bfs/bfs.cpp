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
#define CHUNK_SIZE_UNVISITED 1024
#define THREASHOLD 1000000
// #define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

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

void bfs_bottom_up_step(Graph g, int* distances, int* new_distances,
                        int *frontier_count) {
    int num_nodes = g->num_nodes;
    #pragma omp parallel
    {
        int local_frontier_count = 0;
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

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                int dist_incoming = distances[incoming];
                if (dist_incoming != NOT_VISITED_MARKER && i != incoming) {
                    new_distances[i] = dist_incoming + 1;
                    local_frontier_count++;
                    break;
                }
            }
        }

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

void bfs_bottom_up(Graph graph, solution* sol) {

    int frontier_count = 0;
    int num_nodes = graph->num_nodes;

    int *distances = (int *)malloc(num_nodes * sizeof(int));
    memset(distances, 0xff, num_nodes * sizeof(int));
    int *new_distances = (int *)malloc(num_nodes * sizeof(int));
    memset(new_distances, 0xff, num_nodes * sizeof(int));

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
    memcpy(sol->distances, new_distances, sizeof(int) * num_nodes);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("duration: %.4f\n", end_time - start_time);
#endif
}

void bfs_hybrid_step(Graph g,
                     vertex_set *frontier, vertex_set *new_frontier,
                     int *distances, int *new_distances) {
    int frontier_count = frontier->count;
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

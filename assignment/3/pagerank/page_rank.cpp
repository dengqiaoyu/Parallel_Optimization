/**
 * @file page_rank.cpp
 * @brief this file contains page rank algorithm that can be run i parallel
 * @author Changkai Zhou (zchangka) Qioayu Deng (qdeng)
 */
#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ABS(x) (x) > 0 ? (x) : -(x)
#define CHUNK_SIZE 4096

/**
 * Calculate the page value for given graph.
 * @param g           graph to process (see common/graph.h)
 * @param solution    array of per-vertex vertex scores
 *                    (length of array is num_nodes(g))
 * @param damping     page-rank algorithm's damping parameter
 * @param convergence page-rank algorithm's convergence threshold
 */
void pageRank(Graph g, double* solution, double damping, double convergence) {

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    #pragma omp parallel for schedule(static, CHUNK_SIZE)
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    /* pre-calculate the attribute sum for all of nodes have no out edges */
    double sum_contri_zero_outgoing_old = 0.0;
    double sum_contri_zero_outgoing_new = 0.0;
    #pragma omp parallel for \
    reduction(+:sum_contri_zero_outgoing_new) \
    schedule(static, chunk_size)
    for (int i = 0; i < numNodes; i++) {
        sum_contri_zero_outgoing_new +=
            (outgoing_size(g, i) == 0 ? solution[i] / numNodes : 0);
    }

    double *contri_per_vertex = (double *)malloc(numNodes * sizeof(double));

    int converged = 0;
    while (converged == 0) {
        sum_contri_zero_outgoing_old = sum_contri_zero_outgoing_new;
        sum_contri_zero_outgoing_new = 0.0;
        double global_diff = 0.0;
        /* pre-calculate all nodes' attribution for current iteration */
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < numNodes; i++) {
            int outgoing_size_num = outgoing_size(g, i);
            contri_per_vertex[i] =
                (outgoing_size_num == 0 ? 0 : solution[i] / outgoing_size_num);
        }
        /* add new attribution to new solution */
        #pragma omp parallel for \
        reduction(+:sum_contri_zero_outgoing_new) \
        reduction(+:global_diff) schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; i++) {
            double local_score_old = solution[i];
            double local_score_new = 0.0;
            if (incoming_size(g, i) > 0) {
                const Vertex* start = incoming_begin(g, i);
                const Vertex* end = incoming_end(g, i);
                for (const Vertex* v = start; v != end; v++) {
                    int adj_id = *(int*) v;
                    local_score_new += contri_per_vertex[adj_id];
                }
            }
            local_score_new =
                damping * (local_score_new + sum_contri_zero_outgoing_old)
                + (1 - damping) / numNodes;
            solution[i] = local_score_new;
            /* if node has no out edge, contribute to every node in the graph */
            sum_contri_zero_outgoing_new +=
                (outgoing_size(g, i) == 0 ? local_score_new : 0);
            global_diff += ABS(local_score_new - local_score_old);
        }
        sum_contri_zero_outgoing_new /= numNodes;
        converged = (global_diff < convergence);
    }
    free(contri_per_vertex);
}

#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ABS(x) (x) > 0 ? (x) : -(x)

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence) {

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    /* 418/618 Students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode:

       // initialization: see example code above
       score_old[vi] = 1/numNodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

         score_new[vi] += sum over all nodes vj with no outgoing edges
                            { damping * score_old[vj] / numNodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }
     */
    double sum_contri_zero_outgoing_old = 0.0;
    double sum_contri_zero_outgoing_new = 0.0;
    #pragma omp parallel for reduction(+:sum_contri_zero_outgoing_new) schedule(static)
    for (int i = 0; i < numNodes; i++) {
        if (outgoing_size(g, i) == 0) {
            sum_contri_zero_outgoing_new += solution[i];
        }
    }
    sum_contri_zero_outgoing_new =
        sum_contri_zero_outgoing_new / numNodes;
    // double *contri_zero_outgoing =
    //     (double*) calloc(num_zero_outgoing_vertex, sizeof(double));
    int converged = 0;
    while (converged == 0) {
        sum_contri_zero_outgoing_old = sum_contri_zero_outgoing_new;
        sum_contri_zero_outgoing_new = 0.0;
        int i = 0;
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+:sum_contri_zero_outgoing_new, global_diff) schedule(static)
        for (i = 0; i < numNodes; i++) {
            double local_score_old = solution[i];
            double local_score_new = 0.0;
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            for (const Vertex* v = start; v != end; v++) {
                int adj_id = *(int*) v;
                local_score_new +=
                    solution[adj_id] / (double)outgoing_size(g, adj_id);
            }

            local_score_new =
                damping * (local_score_new + sum_contri_zero_outgoing_old) + (double)(1 - damping) / numNodes;
            // local_score_new += sum_contri_zero_outgoing_old;
            if (outgoing_size(g, i) == 0) {
                sum_contri_zero_outgoing_new += local_score_new;
            }
            solution[i] = local_score_new;
            global_diff += ABS(local_score_new - local_score_old);
        }
        sum_contri_zero_outgoing_new = sum_contri_zero_outgoing_new / numNodes;
        // printf("sum_contri_zero_outgoing: %.12f\n", sum_contri_zero_outgoing);
        printf("global_diff: %.20f\n", global_diff);
        // printf("convergence: %.20f\n", convergence);
        converged = (global_diff < convergence);
    }
    // printf("outgoing_size: %d\n", outgoing_size(g, 2590186));
    // printf("outgoing_size: %d\n", outgoing_size(g, 2590187));
    // printf("outgoing_size: %d\n", outgoing_size(g, 2590188));
    // printf("solution[%d]: %.20f\n", 2590187, solution[2590187]);
    // exit(1);
}

#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


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

    // 1. Computation 2.Parallel, barrier  3.critical + cache

    int numNodes = num_nodes(g);
    bool converged = 0;


    double equal_prob = 1.0 / numNodes;
    int chunksize = 5000;
    double sum_nooutnode_pr_new = 0.0;
    double sum_nooutnode_pr_old = 0.0;
    double* mean_edge_pr;
    mean_edge_pr = (double*)malloc(sizeof(double) * g->num_nodes);

    // Initialization
    #pragma omp parallel for schedule(static, chunksize)
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    #pragma omp parallel for reduction(+:sum_nooutnode_pr_new) schedule(dynamic, chunksize)
    for (int i = 0; i < numNodes; i++) {
        sum_nooutnode_pr_new +=
            (outgoing_size(g, i) == 0 ? solution[i] / numNodes : 0);
    }

    while (!converged) {
        sum_nooutnode_pr_old = sum_nooutnode_pr_new;
        // Cal mean pr value of outgoing nodes for each node

        #pragma omp parallel for schedule(dynamic, chunksize)
        for (int i = 0; i < numNodes; ++i) {
            int outgoing_size_num = outgoing_size(g, i);
            mean_edge_pr[i] =
                (outgoing_size_num == 0 ? 0 : solution[i] / outgoing_size_num);
        }


        double diff = 0.0;
        // Loop all incoming edge
        #pragma omp parallel for reduction(+:diff, sum_nooutnode_pr_new) schedule(dynamic,chunksize)
        for (int i = 0; i < numNodes; ++i) {
            double solution_new_unit = 0.0;
            if (incoming_size(g, i) > 0) {
                const Vertex* start = incoming_begin(g, i);
                const Vertex* end = incoming_end(g, i);

                // [DEBUG] node
                for (const Vertex* node = start; node < end; ++node) {
                    solution_new_unit += mean_edge_pr[*node];
                }
            }

            solution_new_unit = (solution_new_unit + sum_nooutnode_pr_old) * damping + (1.0 - damping) / numNodes;
            sum_nooutnode_pr_new +=
                (outgoing_size(g, i) == 0 ? solution_new_unit : 0);
            diff += fabs(solution_new_unit - solution[i]);
            solution[i] = solution_new_unit;
        }
        sum_nooutnode_pr_new /= numNodes;
        converged = (diff < convergence);
    }
}

//    // 418/618 Students: Implement the page rank algorithm here.  You
//    //   are expected to parallelize the algorithm using openMP.  Your
//    //   solution may need to allocate (and free) temporary arrays.

//    //   Basic page rank pseudocode:

//      // initialization: see example code above
//     score_old[vi] = 1/numNodes;



//     while (!converged) {

//        // compute score_new[vi] for all nodes vi:
//        score_new[vi] = sum over all nodes vj reachable from incoming edges
//                           { score_old[vj] / number of edges leaving vj  }
//        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;




//        // Add a fraction of the leftover mass from all nodes with no outgoing
//        // edges onto this node
//        score_new[vi] += sum over all nodes vj with no outgoing edges
//                           { damping * score_old[vj] / numNodes }

//        // compute how much per-node scores have changed
//        // quit once algorithm has converged

//        global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
//        converged = (global_diff < con
//         vergence)
//      }


// }

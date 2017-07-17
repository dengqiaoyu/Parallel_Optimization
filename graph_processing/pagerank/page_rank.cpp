
#include "page_rank.h"
#include <vector>
#include <omp.h>
// #define DEBUG
#define MASTER 0
#define DEFAULT_TAG 0
#define chunksize 1000

/* Potential Bugs
 * vector assignment, clear, from array -> vector
 * MPI: send/rec working pattern
*/

void pageRank(DistGraph &g, double* solution, double damping,
              double convergence) {

    int totalVertices = g.total_vertices();
    int vertices_per_process = g.vertices_per_process;
    int startVertex = g.start_vertex;
    int endVertex = g.end_vertex + 1;

    int worldRank = g.world_rank;
    std::vector<double> score_curr(totalVertices, 0.0);
    std::vector<double> score_last(totalVertices, 0.0);
    //update to zeros every loop
    std::vector<double> score_next(vertices_per_process, 0.0);

    int converged = 0;

    double equal_prob = 1.0 / totalVertices;

    // initialize per-vertex scores
    // #pragma omp parallel for schedule(auto)
    #pragma omp parallel for schedule(static, chunksize)
    for (int i = 0; i < totalVertices; ++i) {
        score_curr[i] = equal_prob;
        score_last[i] = equal_prob;
    }

    while (!converged) {

        // Find all no-outgoing vertices and cal
        double sum_nooutnode_pr = 0.0;
        double nooutnode_pr = 0.0;

        #pragma omp parallel for schedule(dynamic, chunksize)
        for (int j = 0; j < vertices_per_process; j++) {
            if (g.out_edges_num[j + startVertex] == 0){
                nooutnode_pr += score_last[j + startVertex];
            }
        }

        MPI_Allreduce(&nooutnode_pr, &sum_nooutnode_pr, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sum_nooutnode_pr /= totalVertices;

        // * 1. Cal no-outgoing vertices pr value (Allreduce)
        #pragma omp parallel
        {

            #pragma omp for schedule(static, chunksize)
            for (int j = 0; j < vertices_per_process; j++) {
                score_next[j] = 0.0;
            }
            // * 2. Cal score_next and broadcast (bcast)
            #pragma omp for schedule(dynamic, chunksize/10)
            for (int dst_idx = startVertex; dst_idx < endVertex; ++dst_idx) {
                int esize = g.in_edge_src_size[dst_idx - startVertex];
                for (int idx = 0; idx < esize; idx++) {
                    int src = g.in_edge_src[dst_idx - startVertex][idx];
                    // Edge edge = g.in_edges[j];
                    score_next[dst_idx - startVertex] +=
                        score_last[src] / g.out_edges_num[src];
                }
            }
            double sum_part =
                (sum_nooutnode_pr  * damping + (1.0 - damping))
                / double(totalVertices);

            #pragma omp for schedule(static, chunksize)
            for (int j = 0; j < vertices_per_process; j++) {
                score_next[j] = score_next[j] * damping + sum_part;
            }
        }

        MPI_Allgather(score_next.data(), vertices_per_process,
                      MPI_DOUBLE, score_curr.data(), vertices_per_process,
                      MPI_DOUBLE, MPI_COMM_WORLD);

        double diff = 0.0;

        if (worldRank == MASTER) {
            #pragma omp parallel for reduction(+:diff) \
            schedule(static, chunksize)
            for (int i = 0; i < totalVertices; ++i) {
                diff += fabs(score_last[i] - score_curr[i]);
            }
            converged = (diff < convergence);
        }

        MPI_Bcast(&converged, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        if (!converged)  score_curr.swap(score_last);
    }

    #pragma omp parallel for schedule(static, chunksize)
    for (int j = 0; j < vertices_per_process; j++) {
        solution[j] = score_curr[startVertex + j];
    }
}

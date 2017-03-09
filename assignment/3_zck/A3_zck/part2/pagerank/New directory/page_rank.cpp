
#include "page_rank.h"
#include <vector>

// #define DEBUG
/*
 * pageRank-- 
 *
 * Computes page rank on a distributed graph g
 * 
 * Per-vertex scores for all vertices *owned by this node* (not all
 * vertices in the graph) should be placed in `solution` upon
 * completion.
 */
void pageRank(DistGraph &g, double* solution, double damping, double convergence) {

    int totalVertices = g.total_vertices();
    int vertices_per_process = g.vertices_per_process;
    int worldSize = g.world_size;
    int worldRank = g.world_rank;
    std::vector<double> score_curr(totalVertices, 0);
    std::vector<double> score_next(g.vertices_per_process, 0);

    int converged = 0; 


    double equal_prob = 1.0 / totalVertices;
    int chunksize = 5000;

    double* solution_old;
    solution_old = (double*)malloc(sizeof(double) * totalVertices);

    int nooutnode_end = 0;
    int* nooutnodes;
    nooutnodes = (int*)malloc(sizeof(int) * totalVertices);
    double* mean_edge_pr;
    mean_edge_pr = (double*)malloc(sizeof(double) * totalVertices);

    // initialize per-vertex scores
    #pragma omp parallel for schedule(static, chunksize)
    for (int i = 0; i < totalVertices; ++i) {
        score_curr[i] = equal_prob;
    }

    // Find all nodes without out-going edges
    // #pragma omp parallel for schedule(dynamic, chunksize)
    
    // //[DEBUG] add later
    // for (int i = 0; i < totalVertices; ++i) {
    //     if (outgoing_size(g, i) == 0){
    //         nooutnodes[nooutnode_end] = i;
    //         nooutnode_end ++;
    //     }
    // }

    while (converged < 3) {

    // while (!converged) {    


        // 1. Cal vertices new score using data in self node
        // 2. Send score_next to other Nodes
        // 3. Update its score_curr with score_next
        // 4. Update its score_curr by recv other nodes' result
        // 5. [?] Add No outgoing nodes value

        // Additional Setup
        // [1] g.edges_num, for each vertex, in setup
        // [2]

        MPI_Request* send_reqs = new MPI_Request[worldSize];
        MPI_Status* probe_status = new MPI_Status[worldSize];
        
        // 1. Cal vertices new score using data in self node
        // [?] atomic oper
        
        // auto: another way to realize for loop of vector
        // #pragma omp parallel for //schedule(dynamic, chunksize)
    
        for(auto &edge: g.in_edges) {
            // score_next[edge.dest] += score_curr[edge.src] / g.edges_num[edge.src];
            score_next[edge.dest] += score_curr[edge.src] / 10.0;
        }

        // 2. Send score_next to other Nodes
        for (int i = 0; i < worldSize; i++) {
            if (i != worldRank) {
                // [?] MPI_double
                // [? DEBUG] score_next . int * (simple type)
                MPI_Isend(score_next.data(), score_next.size(), MPI_DOUBLE,
                          i, 0, MPI_COMM_WORLD, &send_reqs[i]);
            }
        }

        // 3. Update its score_curr with score_next
        int start = g.start_vertex;
        int end = g.end_vertex;

        for(int i = 0; i != score_next.size(); i++) {
#ifdef DEBUG
            if (i + start < end) score_curr[i+start] = score_next[i];
#else
            score_curr[i+start] = score_next[i];
#endif                    
        }


        // 4. Update its score_curr by recv other nodes' result
        for (int i = 0; i < worldSize; i++) {
            if (i != worldRank) {
                MPI_Status status;
                // Allow checking of incoming messages, without actual receipt of them. 
                //   the user can then decide how to receive them
                //   based on the information returned by the probe in the status variable
                MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
                int num_vals = 0;
                MPI_Get_count(&probe_status[i], MPI_DOUBLE, &num_vals);

                std::vector<double> recv_score(num_vals);
                // receive message 
                MPI_Recv(recv_score.data(), num_vals, MPI_DOUBLE, probe_status[i].MPI_SOURCE,
                         probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);

                // push the received edge into vector
                // decide whether or not to receive new edges?
                int start = i * vertices_per_process;
                int end = std::min((i+1) * vertices_per_process, totalVertices);

            for(int i = 0; i != recv_score.size(); i++) {
#ifdef DEBUG
                if (i + start < end) score_curr[i+start] = recv_score[i];
#else
                score_curr[i+start] = recv_score[i];
#endif                    
            }

                // free memory of vector
                recv_score.clear();
                std::vector<double>(recv_score).swap(recv_score);
            }
        }        



        // 5. [?] Add No outgoing nodes value
        // Cal mean pr value of outgoing nodes for each node

        // 6. Delete: free memory 
        delete(send_reqs);
        delete(probe_status);

        // 7. Set barrier
        MPI_Barrier(MPI_COMM_WORLD);


        // #pragma omp parallel for schedule(dynamic, chunksize)
        // for (int i = 0; i < totalVertices; ++i) {
        //   if (outgoing_size(g, i) == 0) mean_edge_pr[i] = 0.0;
        //   else mean_edge_pr[i] = solution[i] / outgoing_size(g, i);  
        // }

        // double sum_nooutnode_pr = 0.0;

        // #pragma omp parallel for reduction(+:sum_nooutnode_pr) schedule(dynamic, chunksize)
        // for (int node = 0; node < nooutnode_end; ++node) {
        //   sum_nooutnode_pr += solution[nooutnodes[node]];
        // }
        // sum_nooutnode_pr /= totalVertices;

        // double diff = 0.0;
        // // Loop all incoming edge 
        // #pragma omp parallel for schedule(dynamic,chunksize)
        // for (int i=0; i < totalVertices; ++i){
        //   double solution_new_unit = 0.0;
        //   if (incoming_size(g, i) > 0){
        //     const Vertex* start = incoming_begin(g, i);
        //     const Vertex* end = incoming_end(g, i);
            
        //     // [DEBUG] node
        //     for (const Vertex* node = start; node < end; ++node) {
        //       solution_new_unit += mean_edge_pr[*node];
        //     }           
        //   }
          
        //   solution_new_unit = (solution_new_unit + sum_nooutnode_pr) * damping + (1.0 - damping) / totalVertices;

        //   solution_old[i] = solution[i];
        //   solution[i] = solution_new_unit;
        // } 

        // // Outside loop
        // #pragma omp parallel for reduction(+:diff)
        // for (int i = 0; i < totalVertices; ++i) {
        //   diff += fabs(solution_old[i] - solution[i]);
        // }

        // converged = (diff < convergence);
        converged ++;
    }


    /*

      Repeating basic pagerank pseudocode here for your convenience
      (same as for part 1 of this assignment)

    while (!converged) {

        // compute score_new[vi] for all vertices belonging to this process
        score_new[vi] = sum over all vertices vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / totalVertices;

        score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / totalVertices }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all vertices vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)

        // Note that here, some communication between all the nodes is necessary
        // so that all nodes have the same copy of old scores before beginning the 
        // next iteration. You should be careful to make sure that any data you send 
        // is received before you delete or modify the buffers you are sending.

    }

    // Fill in solution with the scores of the vertices belonging to this node.

    */
}

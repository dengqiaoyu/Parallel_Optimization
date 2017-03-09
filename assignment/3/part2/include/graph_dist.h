#ifndef __DISTGRAPH_DEFINED
#define __DISTGRAPH_DEFINED

#include <mpi.h>
#include <stdio.h>
#include <random>
#include <set>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unistd.h>

#include "graph_dist_ref.h"

using Vertex = int;

/*
 * Representation of a distributed graph.  Each node in the cluster is
 * the "owner" of a subset of the vertices in the graph
 */
class DistGraph {
  public:
    int vertices_per_process;   // vertices per cluster node
    int max_edges_per_vertex;

    int world_size;
    int world_rank;

    // start and end vertex ids of the span of vertices "owned" by the
    // this node.  (nodes own a contiguous span of vertices)
    Vertex start_vertex;
    Vertex end_vertex;

    DistGraph(int _vertices_per_process, int _max_edges_per_vertex,
              GraphType _type, int _world_size, int _world_rank);

    int get_vertex_owner_rank(Vertex v);
    int total_vertices();

    // graph generation routines and helpers
    bool is_left_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_right_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_top_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    bool is_bottom_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process);
    void generate_graph_uniform();
    void generate_graph_grid();
    void generate_graph_clustered();
    void get_incoming_edges(const std::vector<std::vector<Edge>> &edge_scatter);

    // array of incoming edges to vertices owned by the node
    // (in_edges[i].dst should always be local to this node)
    std::vector<Edge> in_edges;
    int **in_edge_src;
    int *in_edge_src_size;

    // array of outgoing edges from vertices owned by the node
    // out_edges[i].src should always be local to this node
    std::vector<Edge> out_edges;
    int **out_edge_dst;
    int *out_edge_dst_size;
    std::vector<int> out_edges_num;


    // Called after in_edges and out_edges are initialized. May be
    // useful for students to precompute/build additional structures
    void setup();
  private:
    void get_vertex_edges_size(int *size_array, int mode);
    void set_edges(int *out_edge_dst_size, int **out_edge_dst,
                   int *in_edge_src_size, int **in_edge_src);
    int get_index(int v);

};

// generates a distributed graph of the given graph type (uniform,
// grid, etc.) and with the provided graph parameters
inline
DistGraph::DistGraph(int _vertices_per_process, int _max_edges_per_vertex,
                     GraphType type, int _world_size, int _world_rank) :
    vertices_per_process(_vertices_per_process),
    max_edges_per_vertex(_max_edges_per_vertex),
    world_size(_world_size),
    world_rank(_world_rank) {
    start_vertex = world_rank * vertices_per_process;
    end_vertex = (world_rank + 1) * vertices_per_process - 1;

    if (type == uniform_random) {
        generate_graph_uniform();
    } else if (type == grid) {
        int closest_sqrt = sqrt(vertices_per_process);
        assert(vertices_per_process == closest_sqrt * closest_sqrt);
        closest_sqrt = sqrt(world_size);
        assert(world_size == closest_sqrt * closest_sqrt);
        generate_graph_grid();
    } else if (type == clustered) {
        generate_graph_clustered();
    } else {
        assert(false);
    }
}

/*
 * get_vertex_owner_rank --
 *
 * Returns the id of the node that is the owner of the vertex
 */
inline
int DistGraph::get_vertex_owner_rank(Vertex v) {
    return (v / vertices_per_process);
}

/*
 * total_vertices --
 *
 * Returns to total number of vertices in the graph
 */
inline
int DistGraph::total_vertices() {
    return vertices_per_process * world_size;
}

/*
 * get_incoming_edges --
 *
 * uses inter-node communication to build a list of in_edges from the
 * distributed list of out_edges
 */
inline
void DistGraph::get_incoming_edges(const std::vector<std::vector<Edge>> &edge_scatter) {
    // printf("begin get_incoming_edges\n");
    /*
    // helpful reminders of MPI send and receive syntax

    int MPI_Isend(void* buf, int count, MPI_Datatype datatype, int dest,
                          int tag, MPI_Comm, comm, MPI_Request *request)

    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
                  int tag, MPI_Comm comm, MPI_Request *request)

    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                 MPI_Comm comm, MPI_Status *status)
    */
    for (auto &e : edge_scatter[world_rank]) {
        in_edges.push_back(e);
    }

    std::vector<int*> send_bufs;
    std::vector<int> send_idx;
    std::vector<int*> recv_bufs;

    MPI_Request* send_reqs = new MPI_Request[world_size];
    MPI_Status* probe_status = new MPI_Status[world_size];

    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            int* send_buf = new int[edge_scatter[i].size() * 2];

            send_bufs.push_back(send_buf);
            send_idx.push_back(i);

            for (size_t j = 0; j < edge_scatter[i].size(); j++) {
                send_buf[2 * j] = edge_scatter[i][j].src;
                send_buf[2 * j + 1] = edge_scatter[i][j].dest;
            }

            MPI_Isend(send_buf, edge_scatter[i].size() * 2, MPI_INT,
                      i, 0, MPI_COMM_WORLD, &send_reqs[i]);
        }
    }

    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
            int num_vals = 0;
            MPI_Get_count(&probe_status[i], MPI_INT, &num_vals);

            int* recv_buf = new int[num_vals];
            recv_bufs.push_back(recv_buf);

            MPI_Recv(recv_buf, num_vals, MPI_INT, probe_status[i].MPI_SOURCE,
                     probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);

            for (int j = 0; j < num_vals; j += 2) {
                assert(get_vertex_owner_rank(recv_buf[j + 1]) == world_rank);
                in_edges.push_back({recv_buf[j], recv_buf[j + 1]});
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
    // printf("end get_incoming_edges\n");
}


inline
void DistGraph::generate_graph_uniform() {
    // Each process creates a set of nodes and outgoing edges
    std::default_random_engine g(world_rank);

    std::uniform_int_distribution<int> edge_dist(1, max_edges_per_vertex);
    std::uniform_int_distribution<int> dest_dist(0, total_vertices() - 1);

    std::vector<std::vector<Edge>> edge_scatter(world_size);

    for (int v = start_vertex; v <= end_vertex; v++) {
        int num_edges = edge_dist(g);
        if (num_edges == 0) {
            std::cout << "UM NO EDGES = " << v << std::endl;
        }
        std::set<int> done;
        for (int e = 0; e < num_edges; e++) {

            int dest = dest_dist(g);
            if (done.find(dest) != done.end()) {
                e--;
                continue;
            }

            done.insert(dest);
            out_edges.push_back({v, dest});

            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
    }

    get_incoming_edges(edge_scatter);
}

/*
 * The following routines are used as helpers in grid graph construction
 */
inline
bool DistGraph::is_left_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process) {
    if (world_rank % sqrt_world_size == 0 && v % sqrt_per_process == 0)
        return true;
    return false;
}

inline
bool DistGraph::is_right_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process) {
    if (world_rank % sqrt_world_size == sqrt_world_size - 1
            && v % sqrt_per_process == sqrt_per_process - 1)
        return true;
    return false;
}

inline
bool DistGraph::is_top_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process) {
    if (world_rank / sqrt_world_size == 0 && (v - start_vertex) / sqrt_per_process == 0)
        return true;
    return false;
}

inline
bool DistGraph::is_bottom_edge_vertex(Vertex v, int sqrt_world_size, int sqrt_per_process) {
    if (world_rank / sqrt_world_size == sqrt_world_size - 1 &&
            (v - start_vertex) / sqrt_per_process == sqrt_per_process - 1)
        return true;
    return false;
}

inline
void DistGraph::generate_graph_grid() {
    // Each process creates a set of nodes and outgoing edges
    int sqrt_per_process = sqrt(vertices_per_process);
    int sqrt_world_size = sqrt(world_size);

    std::vector<std::vector<Edge>> edge_scatter(world_size);

    for (int v = start_vertex; v <= end_vertex; v++) {
        // For the edges of the grid, don't add vertices outside the grid
        int dest;
        if (!is_left_edge_vertex(v, sqrt_world_size, sqrt_per_process)) {
            if (v % sqrt_per_process == 0) {
                dest = v - (vertices_per_process - sqrt_per_process + 1);
            } else {
                dest = v - 1;
            }
            out_edges.push_back({v, dest});
            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
        if (!is_right_edge_vertex(v, sqrt_world_size, sqrt_per_process)) {
            if (v % sqrt_per_process == sqrt_per_process - 1) {
                dest = v + (vertices_per_process - sqrt_world_size + 1);
            } else {
                dest = v + 1;
            }
            out_edges.push_back({v, dest});
            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
        if (!is_top_edge_vertex(v, sqrt_world_size, sqrt_per_process)) {
            if ((v - start_vertex) / sqrt_per_process == 0) {
                dest = v - ((sqrt_world_size - 1) * vertices_per_process + sqrt_per_process);
            } else {
                dest = v - sqrt_per_process;
            }
            out_edges.push_back({v, dest});
            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
        if (!is_bottom_edge_vertex(v, sqrt_world_size, sqrt_per_process)) {
            if ((v - start_vertex) / sqrt_per_process == sqrt_per_process - 1) {
                dest = v + ((sqrt_world_size - 1) * vertices_per_process + sqrt_per_process);
            } else {
                dest = v + sqrt_per_process;
            }
            out_edges.push_back({v, dest});
            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
    }

    get_incoming_edges(edge_scatter);
}

inline
void DistGraph::generate_graph_clustered() {
    // Each process creates a set of nodes and outgoing edges
    std::default_random_engine g(world_rank);

    std::uniform_int_distribution<int> edge_dist(1, max_edges_per_vertex);
    std::uniform_int_distribution<int> dest_dist(start_vertex, end_vertex - 1);
    std::uniform_int_distribution<int> dest_foreign_dist(0, total_vertices() - 1);
    std::uniform_real_distribution<double> ratio(0.0, 1.0);

    std::vector<std::vector<Edge>> edge_scatter(world_size);

    for (int v = start_vertex; v <= end_vertex; v++) {
        int num_edges = edge_dist(g);
        std::set<int> done;
        for (int e = 0; e < num_edges; e++) {
            // Toss a coin and with CLUSTERING_RATIO probability, pick an internal edge
            int dest;
            if ( ratio(g) <= CLUSTERING_RATIO )
                dest = dest_dist(g);
            else
                dest = dest_foreign_dist(g);

            if (done.find(dest) != done.end()) {
                e--;
                continue;
            }

            done.insert(dest);
            out_edges.push_back({v, dest});

            edge_scatter[get_vertex_owner_rank(dest)].push_back({v, dest});
        }
    }

    get_incoming_edges(edge_scatter);
}

/*
 * setup --
 */
inline
void DistGraph::setup() {
    // printf("begin setup\n");

    int out_edge = 0;
    int in_edge = 1;
    // This method is called after in_edges and out_edges
    // have been initialized.  This is the point where student code may wish
    // to setup its data structures, precompute anythign about the
    // topology, or put the graph in the desired form for future computation.
    out_edge_dst = (int **)malloc(vertices_per_process * sizeof(int *));
    in_edge_src = (int **)malloc(vertices_per_process * sizeof(int *));
    out_edge_dst_size = (int *)calloc(vertices_per_process, sizeof(int));
    in_edge_src_size = (int *)calloc(vertices_per_process, sizeof(int));

    /* get the edges size for every node in local */
    get_vertex_edges_size(out_edge_dst_size, out_edge);
    get_vertex_edges_size(in_edge_src_size, in_edge);

    for (int v = start_vertex; v <= end_vertex; v++) {
        int idx = v - start_vertex;
        out_edge_dst[idx] = (int *)malloc(out_edge_dst_size[idx] * sizeof(int));
        in_edge_src[idx] = (int *)malloc(in_edge_src_size[idx] * sizeof(int));
    }

    /* set local nodes' edges according to out_edge and in_edge */
    set_edges(out_edge_dst_size, out_edge_dst, in_edge_src_size, in_edge_src);
    out_edges_num.reserve(total_vertices());
    std::vector<int> out_edges_num_part(vertices_per_process, 0);

    for (auto &edge : out_edges) {
        out_edges_num_part[edge.src - start_vertex] += 1;
    }

    MPI_Allgather(out_edges_num_part.data(), vertices_per_process,
                  MPI_INT, out_edges_num.data(), vertices_per_process,
                  MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

inline
void DistGraph::get_vertex_edges_size(int *size_array, int mode) {
    if (mode == 0) {
        for (auto &e : out_edges) {
            // printf("%d->%d\n", e.src, e.dest);
            size_array[e.src - start_vertex]++;
        }
    } else {
        for (auto &e : in_edges) {
            size_array[e.dest - start_vertex]++;
        }
    }
}

inline
void DistGraph::set_edges(int *out_edge_dst_size, int **out_edge_dst,
                          int *in_edge_src_size, int **in_edge_src) {
    int *size_idx_array = (int *)calloc(vertices_per_process, sizeof(int));
    for (auto &e : out_edges) {
        int idx = e.src - start_vertex;
        out_edge_dst[idx][size_idx_array[idx]++] = e.dest;
    }
    memset(size_idx_array, 0, vertices_per_process * sizeof(int));
    for (auto &e : in_edges) {
        int idx = e.dest - start_vertex;
        in_edge_src[idx][size_idx_array[idx]++] = e.src;
    }
}

inline
int DistGraph::get_index(int v) {
    return v - start_vertex;
}

#endif

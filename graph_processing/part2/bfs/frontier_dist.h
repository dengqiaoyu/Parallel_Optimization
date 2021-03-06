#include <cstring>

using Vertex = int;
#define FRONTIER_TAG 0
#define EMPTY_TAG 1

/**
 * Class that stores a distributed frontier. Each node has world_size arrays, one
 * dedicated to each node. Add populates these arrays locally, and sync merges
 * them so that the local frontier for each node (containing only local vertices)
 * is present on that node.
 */
class DistFrontier {
  public:
    // Maximum number of vertices that a single node's frontier could have
    // at any given point in time
    int max_vertices_per_node;

    // Distributed frontier structure - every node independently produces a new
    // frontier using its local vertices, and places the frontier vertices in the
    // arrays corresponding to the owning nodes for each destination.
    //
    // For example: elements[2] constains all the frontier vertices that are owned
    // by process 2
    Vertex **elements;
    int **depths;
    int *sizes;

    int world_size;
    int world_rank;

    DistFrontier(int _max_vertices_per_node, int _world_size, int _world_rank);
    ~DistFrontier();

    void clear();
    void add(int owner_rank, Vertex v, int depth);

    int get_local_frontier_size();
    int *get_local_frontier_size_ptr();
    Vertex* get_local_frontier();
    int *get_local_depths();

    int get_remote_frontier_size(int remote_rank);
    int *get_remote_frontier(int remote_rank);
    int *get_remote_depths(int remote_rank);

    bool is_empty(int iterate);
};

inline
DistFrontier::DistFrontier(int _max_vertices_per_node, int _world_size,
                           int _world_rank) :
    max_vertices_per_node(_max_vertices_per_node),
    world_size(_world_size),
    world_rank(_world_rank) {
    elements = new Vertex*[world_size];
    depths = new int*[world_size];
    for (int i = 0; i < world_size; ++i) {
        elements[i] = new Vertex[max_vertices_per_node];
        depths[i] = new int[max_vertices_per_node];
    }

    sizes = new int[world_size]();
}

inline
DistFrontier::~DistFrontier() {
    if (elements) {
        for (int i = 0; i < world_rank; ++i) {
            if ( elements[i] ) delete elements[i];
            if ( depths[i] ) delete depths[i];
        }

        delete elements;
        if (depths) delete depths;
    }

    if (sizes) delete sizes;
}

inline
void DistFrontier::clear() {
    memset(sizes, 0, world_size * sizeof(int));
}

inline
void DistFrontier::add(int owner_rank, Vertex v, int depth) {
    elements[owner_rank][sizes[owner_rank]] = v;
    depths[owner_rank][sizes[owner_rank]++] = depth;
}

inline
int DistFrontier::get_local_frontier_size() {
    return sizes[world_rank];
}

inline
int *DistFrontier::get_local_frontier_size_ptr() {
    return sizes + world_rank;
}

inline
Vertex* DistFrontier::get_local_frontier() {
    return elements[world_rank];
}

inline
Vertex* DistFrontier::get_local_depths() {
    return depths[world_rank];
}

inline
int DistFrontier::get_remote_frontier_size(int remote_rank) {
    return sizes[remote_rank];
}

inline
int *DistFrontier::get_remote_frontier(int remote_rank) {
    return elements[remote_rank];
}

inline
int *DistFrontier::get_remote_depths(int remote_rank) {
    return depths[remote_rank];
}

/**
 * [DistFrontier::is_empty description]
 * @param  iterate the current iteration, for debug using
 * @return         true for covered, false for not ready converged
 */
inline
bool DistFrontier::is_empty(int iterate) {
    MPI_Request* send_reqs = new MPI_Request[world_size];
    bool is_empty = true;

    /* check if all the frontier in local has zero node */
    for (int rank = 0; rank < world_size; rank++) {
        if (get_remote_frontier_size(rank) != 0) {
            is_empty = false;
            break;
        }
    }
    int flag = (is_empty == true ? 1 : 0);

    /**
     * send check result to every machine in the network, if one of them does
     * have non empty frontier, then we cannot stop
     */
    for (int rank = 0; rank < world_size; rank++) {
        if (rank == world_rank)
            continue;
        MPI_Isend(&flag, 1, MPI_INT, rank, EMPTY_TAG,
                  MPI_COMM_WORLD, &send_reqs[rank]);
    }

    for (int rank = 0; rank < world_size; rank++) {
        int recv_buf;
        if (rank == world_rank)
            continue;
        MPI_Status status;
        MPI_Recv(&recv_buf, 1, MPI_INT, rank, EMPTY_TAG,
                 MPI_COMM_WORLD, &status);
        if (recv_buf == 0)
            is_empty = false;
    }

    for (int rank = 0; rank < world_size; rank++) {
        if (rank == world_rank)
            continue;
        MPI_Status status;
        MPI_Wait(&send_reqs[rank], &status);
    }
    delete(send_reqs);

    return is_empty;
}


// Support different synchronization strategies 
// NO synchronization at all
// Synchronization via global locking of data structure
// Basic use of fine-grained locks
// Full version of fine-grained locks
typedef enum { NO_SYNCH, GLOBAL_SYNCH, SIMPLE_SYNCH, FULL_SYNCH } synch_t;

// Nodes in BST
struct Node {
    Node *left, *right;
    Lock lock;
    int value;

    Node(int value);
};

// BST representation
class BST {
 public:
    mode_t mode;              // What synchronization mechanisms to use
    int count;                // How many elements are in the tree

    BST(mode_t mode);         // Create empty BST
    bool insert(int val);     // Insert value into BST
    bool remove(int val);     // Remove a value from BST
    int sum();                // Compute the sum of the elements
    void print(bool show_tree); // Print the members

 private:              // Representation
    Lock glock;        // Protect global data
    // Have dummy node, such that true root is its left child
    Node *pseudoroot;
    // Helper routines

    bool insert_basic(int val);
    bool insert_simple_synch(int val);
    bool insert_full_synch(int val);

    bool remove_basic(int val);
    bool remove_full_synch(int val);

    int sum_subtree_basic(Node *node);
    int sum_subtree_simple_synch(Node *node);
    int sum_subtree_full_synch(Node *node, Node *parent);

    void print_subtree(Node *n, bool show_tree);
};


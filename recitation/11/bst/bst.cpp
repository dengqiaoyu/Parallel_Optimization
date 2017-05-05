#include "tsynch.h"
#include "bst.h"

// Construct node with no children
Node::Node(int v) {
    value = v;
    left = right = NULL;
}

// Create empty BST
BST::BST(mode_t m) {
    mode = m;
    pseudoroot = new Node(-1); // Actual root at pseudoroot->left
    count = 0;
}

// Insert value into BST.  Return false if it's already there
// Sequential code
bool BST::insert_basic(int val) {
    // Strategy: Find position at which to insert node
    Node **nloc = &pseudoroot->left;  // Possible insertion address
    Node *node = *nloc;               // Current node
    bool found = false;
    while (node != NULL) {
	if (val == node->value) {
	    // Already in tree
	    found = true;
	    break;
	} else {
	    // Walk down tree to find insertion point
	    if (val <= node->value)
		nloc = &node->left;
	    else 
		nloc = &node->right;
	    node = *nloc;
	}
    }
    if (!found) {
	// Insert new node
	*nloc = new Node(val);
	count++;
    }
    return !found;
}

// Insert value into BST.  Return false if it's already there
// Locking good enough to allow multiple insertions concurrently 
bool BST::insert_simple_synch(int val) {
    pseudoroot->lock.lock();
    Node *parent = pseudoroot;       // Parent of node being examined
    Node **nloc = &pseudoroot->left; // Insertion position within parent
    Node *node = *nloc;              // Node being examined
    while (node != NULL) {
	// NOT hand-over-hand
	parent->lock.unlock();       
	node->lock.lock();
	if (val == node->value) {
	    // Already in tree
	    node->lock.unlock();
	    return false;
	} else {
	    parent = node;
	    if (val <= node->value)
		nloc = &node->left;
	    else 
		nloc = &node->right;
	    node = *nloc;
	}
    }
    *nloc = new Node(val);
    parent->lock.unlock();
    glock.lock();
    count++;
    glock.unlock();
    return true;
}

// Insert value into BST.  Return false if it's already there
// Locking good enough to allow concurrent insertions, deletions, and sums
bool BST::insert_full_synch(int val) {
    pseudoroot->lock.lock();
    Node *parent = pseudoroot;
    Node **nloc = &pseudoroot->left;
    Node *node = *nloc;
    while (node != NULL) {
	// Hand-over-hand locking
	node->lock.lock();
	parent->lock.unlock();
	if (val == node->value) {
	    // Already in tree
	    node->lock.unlock();
	    return false;
	} else {
	    parent = node;
	    if (val <= node->value)
		nloc = &node->left;
	    else 
		nloc = &node->right;
	    node = *nloc;
	}
    }
    *nloc = new Node(val);
    parent->lock.unlock();
    glock.lock();
    count++;
    glock.unlock();
    return true;
}

// Top level insertion code
bool BST::insert(int val) {
    bool result = false;
    switch (mode) {
    case GLOBAL_SYNCH:
	glock.lock();
    case NO_SYNCH:
	result = insert_basic(val);
	if (mode == GLOBAL_SYNCH)
	    glock.unlock();
	break;
    case SIMPLE_SYNCH:
	result = insert_simple_synch(val);
	break;
    case FULL_SYNCH:
	result = insert_full_synch(val);
	break;
    default:
	report(0, "Unknown mode %d", mode);
	break;
    }
    return result;
}
    
// Remove value from BST
// Return false if value not present
// Removal can require restructuring of key
bool BST::remove_basic(int val) {
    // Want to find position in tree currently holding val
    Node **nloc = &pseudoroot->left;  // Location in parent
    Node *node = *nloc;               // of node being examined
    // Traverse tree to find node
    while (node != NULL && node->value != val) {
	if (val <= node->value)
	    nloc = &node->left;
	else
	    nloc = &node->right;
	node = *nloc;
    }
    if (node == NULL)
	// Value not in tree
	return false;
    // Value to be deleted is in node
    if (node->left == NULL || node->right == NULL) {
	// Easy case: one of node's subtrees is empty
	// Can replace node by the subtree
	*nloc = node->left ? node->left : node->right;
	// Free node that formely held val
	delete node;
    } else {
	// Node has two subtrees
	// Locate minimum element in right subtree
	Node **mloc = &node->right; // Location in parent
	Node *mnode = *mloc;        // of possible minimum element
	while (mnode->left != NULL) {
	    mloc = &(mnode->left);
	    mnode = *mloc;
	}
	// mnode value becomes new value for element
	node->value = mnode->value;
	// replace pointer to mnode in parent with mnode's right subtree
	*mloc = mnode->right;
	// Free node that formerly held minimum value
	delete mnode;
    }
    count--;
    return true;
}

// Remove value from BST
// Return false if value not present
// Removal can require restructuring of key
// Support locking that can run concurrently with inserts and deletes
bool BST::remove_full_synch(int val) {
    pseudoroot->lock.lock();
    Node *parent = pseudoroot;
    Node **nloc = &pseudoroot->left;
    Node *node = *nloc;
    if (node) 
	node->lock.lock();
    while (node != NULL && node->value != val) {
	// Hand-over-hand locking, but maintain 2 levels of locks
	// One on node, and one on its parent
	if (val <= node->value) {
	    nloc = &node->left;
	} else {
	    nloc = &node->right;
	}
	parent->lock.unlock();
	parent = node;
	node = *nloc;
	if (node) 
	    node->lock.lock();
    }
    if (node == NULL) {
	// Value not in tree.  Only need to release parent lock
	parent->lock.unlock();
	return false;
    }
    // Value to be deleted is in node node
    // nloc points to location in tree holding pointer to node
    // Have lock of node and its parent
    if (node->left == NULL || node->right == NULL) {
	// One of node's subtrees is empty
	// Replace pointer to node by pointer to other subtree
	*nloc = node->left  ? node->left : node->right;
	parent->lock.unlock();
	// Don't really need to do this, but it's a good habit
	node->lock.unlock();
	delete node;
    } else {
	// Node has two subtrees.  Must find node with minimum
	// value in right subtree.  
	// Can remove lock on parent, but keep lock on node
	parent->lock.unlock();
	// Keep two levels locked
	parent = NULL;  // Real parent is node, but it's already locked
	Node **mloc = &node->right;
	Node *mnode = *mloc;
	mnode->lock.lock();
	while (mnode->left != NULL) {
	    // Hand-over-hand locking as find node with minimum value
	    if (parent)
		parent->lock.unlock();
	    parent = mnode;
	    mloc = &(mnode->left);
	    mnode = *mloc;
	    mnode->lock.lock();
	}
	node->value = mnode->value;
	*mloc = mnode->right;
	if (parent)
	    parent->lock.unlock();
	node->lock.unlock();
	// Don't really need to do this, but it's a good habit
	mnode->lock.unlock();
	delete mnode;
    }
    glock.lock();
    count--;
    glock.unlock();
    return true;
}

bool BST::remove(int val) {
    bool result = false;
    switch (mode) {
    case GLOBAL_SYNCH:
	glock.lock();
    case NO_SYNCH:
    case SIMPLE_SYNCH:
	result = remove_basic(val);
	if (mode == GLOBAL_SYNCH)
	    glock.unlock();
	break;
    case FULL_SYNCH:
	result = remove_full_synch(val);
	break;
    default:
	report(0, "Unknown mode %d", mode);
	break;
    }
    return result;
}

int BST::sum_subtree_basic(Node *n) {
    if (n == NULL)
	return 0;
    return n->value
	+ sum_subtree_basic(n->left)
	+ sum_subtree_basic(n->right);
}


// Synchronization good enough to be concurrent with insertions
int BST::sum_subtree_simple_synch(Node *node) {
    // If node nonnull, then it is locked
    // and should be released by this function
    if (node == NULL)
	return 0;

    Node *left = node->left;
    Node *right = node->right;
    if (left)
	left->lock.lock();
    if (right)
	right->lock.lock();
    node->lock.unlock();

    int val = node->value;
    val += sum_subtree_simple_synch(left);
    val += sum_subtree_simple_synch(right);

    return val;
}

// Called at top level with node = root, parent = pseudoroot
int BST::sum_subtree_full_synch(Node *node, Node *parent) {
    // If parent nonnull, then it is locked and should be released
    if (node == NULL) {
	if (parent)
	    parent->lock.unlock();
	return 0;
    }
    node->lock.lock();    
    if (parent)
	parent->lock.unlock();
    int val = node->value;
    val += sum_subtree_full_synch(node->left, NULL);
    // Can release lock on this node once start summing right subtree
    val += sum_subtree_full_synch(node->right, node);
    return val;
}

int BST::sum() {
    int val = 0;
    Node *root = pseudoroot->left;
    switch (mode) {
    case NO_SYNCH:
	val = sum_subtree_basic(root);
	break;
    case SIMPLE_SYNCH:
	// Perform lock sequence to ensure can't overtake other operations
	pseudoroot->lock.lock();
	if (root)
	    root->lock.lock();
	pseudoroot->lock.unlock();
	if (!root)
	    return 0;
	val = sum_subtree_simple_synch(root);
	break;
    case GLOBAL_SYNCH:
	glock.lock();
	val = sum_subtree_basic(root);
	glock.unlock();
	break;
    case FULL_SYNCH:
	pseudoroot->lock.lock();
	val = sum_subtree_full_synch(root, pseudoroot);
	break;
    default:
	report(0, "Unknown mode %d", mode);
    }
    return val;
}




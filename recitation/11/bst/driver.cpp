#include "tsynch.h"
#include "bst.h"
#include "driver.h"

/* Driver support */
int maxsize = 0;
BST *tree = NULL;
ISet *iset = NULL;
bool insertOnly = false;
double sumProbability = 0.0;


void init_driver(mode_t mode, int msize, double sprob, bool ionly) {
    maxsize = msize;
    insertOnly = ionly;
    tree = new BST(mode);
    iset = new ISet(2*maxsize);
    sumProbability = sprob;
}

void free_driver() {
    if (insertOnly) {
	tree->mode = NO_SYNCH;
	printf("Final tree:");
	tree->print(true);
    }
    delete tree;
    delete iset;
}

void drive() {
    bool sum = choose_with_probability(sumProbability);
    double insert_prob = insertOnly ? 1.0 : (double )(maxsize - tree->count)/maxsize;
    bool insert = !sum && choose_with_probability(insert_prob);
    if (sum) {
	report(2, "Computing sum");
	int val = tree->sum();
	report(0, "Sum = %d", val);
    } else if (insert) {
	int val = iset->addSomeVal();
	report(2, "Inserting %d", val);
	if (tree->insert(val)) {
	    report(1, "Inserted %d into tree", val);
	    if (!insertOnly)
		tree->print(true);
	} else {
	    report(0, "Could not insert %d into tree", val);
	}
    } else {
	int val = iset->removeSomeVal();
        report(2, "Removing %d", val);
	if (val >= 0) {
	    if (tree->remove(val)) {
		report(1, "Removed value %d from tree", val);
		tree->print(true);
	    } else {
		report(0, "Couldn't find Value %d the tree", val);
	    }
	} else
	    report(1, "Tree empty.  Couldn't remove anything");
    }
}

void BST::print_subtree(Node *n, bool show_tree) {
    if (n == NULL)
	return;
    if (show_tree)
	printf("[");
    print_subtree(n->left, show_tree);
    if (show_tree)
	printf(" %d ", n->value);
    else
	printf(" %d", n->value);	
    print_subtree(n->right, show_tree);
    if (show_tree)
	printf("]");
}

void BST::print(bool show_tree) {
    if (!show_tree)
	printf("[");
    print_subtree(pseudoroot->left, show_tree);
    if (!show_tree)
	printf("]");
    printf("\n");
}


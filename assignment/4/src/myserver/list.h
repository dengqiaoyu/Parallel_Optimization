/**
 * @file list.c
 * @brief this file contains list function declaretion and structure definition.
 * @author Newton Xie(ncx), Qiaoyu Deng(qdeng)
 * @bug No known bugs
 */

#ifndef __LIST_H__
#define __LIST_H__

#define SUCCESS 0
#define ERROR_INIT_LIST_CALLOC_FAILED -1

typedef struct node node_t;
struct node {
    node_t *prev;
    node_t *next;
    char data[0];
};

typedef struct list list_t;
struct list {
    int node_cnt;
    node_t *head;
    node_t *tail;
};

/**
 * Initailize the list.
 * @param  list the pointer to the list
 * @return      SUCCESS(0) for success, ERROR_INIT_LIST_CALLOC_FAILED(-1) for
 *              fail
 */
int list_init(list_t *list);

/**
 * Add new node after the dummy head node.
 * @param list the pointer to the list
 * @param node the pointer to the node
 */
void add_node_to_head(list_t *list, node_t *node);

/**
 * Add new node after the dummy tail node.
 * @param list the pointer to the list
 * @param node the pointer to the node
 */
void add_node_to_tail(list_t *list, node_t *node);

/**
 * Delete node list.
 * @param list the pointer to the list
 * @param node the pointer to the node
 */
void delete_node(list_t *list, node_t *node);

/**
 * Get the first node's in the list(not the head node)
 * @param  list the pointer to the list
 * @return      the pointer to the node for success, NULL for fail
 */
node_t *get_first_node(list_t *list);

/**
 * Get the last node's in the list(not the tail node)
 * @param  list the pointer to the list
 * @return      the pointer to the node for success, NULL for fail
 */
node_t *get_last_node(list_t *list);

/**
 * Get the first node in the list and unlinked it in the list.
 * @param  list the pointer to the list
 * @return      the pointer to the node for success, NULL for fail
 */
node_t *pop_first_node(list_t *list);

/**
 * Get the last node in the list and unlinked it in the list.
 * @param  list the pointer to the list
 * @return      the pointer to the node for success, NULL for fail
 */
node_t *pop_last_node(list_t *list);

void delete_link(list_t *list, node_t *node);

/**
 * Clear all the nodes in the list.
 * @param list the pointer to the list
 */
void clear_list(list_t *list);

#endif /* __LIST_H__ */

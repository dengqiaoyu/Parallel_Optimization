#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

#include <stdlib.h>
#include <map>
#include "list.h"
#include "server/messages.h"

#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

typedef struct cached_item {
    std::string clt_req;
    Response_msg resp;
} cached_item_t;

typedef struct lru_cache {
    list_t lru_list;
    std::map<std::string, node_t *> resp_map;
    int curr_size;
    int max_size;
} lru_cache_t;

void cache_init(lru_cache_t &lru_cache, int max_size);
void cache_put(lru_cache_t &lru_cache,
               std::string &clt_req,
               const Response_msg &resp);
int cache_get(lru_cache_t &lru_cache,
              std::string &clt_req,
              Response_msg &resp);
void evict_one(lru_cache_t &lru_cache);

void cache_init(lru_cache_t &lru_cache, int max_size) {
    list_init(&lru_cache.lru_list);
    lru_cache.curr_size = 0;
    lru_cache.max_size = max_size;
}

void cache_put(lru_cache_t &lru_cache,
               std::string &clt_req,
               const Response_msg &resp) {
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    if (lru_cache.curr_size == lru_cache.max_size) {
        evict_one(lru_cache);
    }
    lru_cache.curr_size++;
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);

    node_t *node = (node_t *)malloc(sizeof(node_t) + sizeof(cached_item_t));
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    cached_item_t *it = (cached_item_t *)(node->data);
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    it->clt_req = clt_req;
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    it->resp = resp;
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    lru_cache.resp_map[clt_req] = node;
    DEBUG_PRINT("line %d in cache_put\n", __LINE__);
    add_node_to_head(&lru_cache.lru_list, node);
}

int cache_get(lru_cache_t &lru_cache,
              std::string &clt_req,
              Response_msg &resp) {
    if (lru_cache.resp_map.find(clt_req) == lru_cache.resp_map.end()) return 0;
    node_t *node = lru_cache.resp_map[clt_req];
    cached_item_t *it = (cached_item_t *)(node->data);
    resp = it->resp;
    delete_link(&lru_cache.lru_list, node);
    add_node_to_head(&lru_cache.lru_list, node);
    return 1;
}

void evict_one(lru_cache_t &lru_cache) {
    lru_cache.curr_size--;
    node_t *node = pop_last_node(&lru_cache.lru_list);
    cached_item_t *it = (cached_item_t *)(node->data);
    lru_cache.resp_map.erase(it->clt_req);
    free(node);
}

#endif /* _LRU_CACHE_H_ */
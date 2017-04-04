#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

#include <stdlib.h>
#include <map>
#include "list.h"
#include "server/messages.h"


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

#endif /* _LRU_CACHE_H_ */
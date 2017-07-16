/**
 * @file This file contains general implementation for LRU cache.
 * @author Qiaoyu Deng(qdeng)
 * @bug No know bugs
 */

#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

#include <stdlib.h>
#include <map>
#include <list>

/* user defined includes */
#include "server/messages.h"

template<typename key_t, typename value_t>
class lru_cache {
  public:
    typedef typename std::pair<key_t, value_t> key_value_pair_t;
    typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

    lru_cache(size_t max_size):
        _max_size(max_size) {}

    void put(const key_t& key, const value_t& value) {
        auto map_it = _cache_map.find(key);
        /* put the new object in the very front of list */
        _cache_lru_list.push_front(key_value_pair_t(key, value));
        /* if this key is already mapped, erase the old one. */
        if (map_it != _cache_map.end()) {
            _cache_lru_list.erase(map_it->second);
            _cache_map.erase(map_it);
        }
        /* map this key to new object */
        _cache_map[key] = _cache_lru_list.begin();
        if (_cache_map.size() > _max_size) {
            /* if the max size of cache is reached, delete the last element */
            auto last = _cache_lru_list.end();
            /* return to the last */
            last--;
            _cache_lru_list.pop_back();
            _cache_map.erase(last->first);
        }
    }

    const value_t& get(const key_t& key) {
        auto map_it = _cache_map.find(key);
        if (map_it == _cache_map.end()) {
            // map not found
        }
        /* put this object at the front of the list */
        _cache_lru_list.splice(_cache_lru_list.begin(),
                               _cache_lru_list,
                               map_it->second);
        return map_it->second->second;
    }

    bool exist(const key_t& key) {
        return (_cache_map.find(key) != _cache_map.end());
    }

    size_t size() {
        return _max_size;
    }
  private:
    std::list<key_value_pair_t> _cache_lru_list;
    std::map<key_t, list_iterator_t> _cache_map;
    size_t _max_size;
};

#endif /* _LRU_CACHE_H_ */
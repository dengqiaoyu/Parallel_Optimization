#ifndef _WORKER_H_
#define _WORKER_H_

// #include <vector>
#include <pthread.h>
#include "server/messages.h"
#include "server/worker.h"
#include "tools/work_queue.h"
#include "return_error.h"
#include "request_type_def.h"

#define DEFAULT_COMPLEXITY 1000000
#define SCHEDULER_LENGTH 64

// #define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

typedef struct fifo_queue_item {
    int complexity;
    int req_type;
    Request_msg req;
} fifo_queue_item_t;

typedef struct fifo_queue {
    pthread_mutex_t fifo_lock;
    int item_cnt;
    WorkQueue<fifo_queue_item_t> queue;
} fifo_queue_t;

typedef struct sche_queue {
    pthread_mutex_t sche_lock;
    int item_cnt;
    int if_being_updated;
    pthread_mutex_t updated_lock;
    WorkQueue<Request_msg> queue;
} sche_queue_t;

typedef struct fast_queue {
    int tellmenow_req_cnt;
    pthread_mutex_t tellmenow_mutex;
    int projectidea_req_cnt;
    pthread_mutex_t projectidea_mutex;
    WorkQueue<Request_msg> tellmenow_queue;
    WorkQueue<Request_msg> projectidea_queue;
} fast_queue_t;

int fifo_queue_init(fifo_queue_t *fifo_queue);
void fifo_queue_put(fifo_queue_t *fifo_queue,
                    const Request_msg &req,
                    int req_type);
int fifo_queue_get(fifo_queue_t *fifo_queue, fifo_queue_item_t &item);

int fast_queue_init(fast_queue_t *fast_queue);
void fast_queue_put(fast_queue_t *fast_queue,
                    const Request_msg &req,
                    int req_type);
int fast_queue_get(fast_queue_t *fast_queue,
                   Request_msg& req,
                   int req_type,
                   int &wait_if_zero,
                   pthread_cond_t *cond);

int sche_queue_init(sche_queue_t *sche_queue);
int sche_queue_get(sche_queue_t *sche_queue,
                   Request_msg &req,
                   fifo_queue_t *fifo_queue);

fifo_queue_item_t create_fifo_queue_item(int complexity,
        const Request_msg& req);
int fill_sche_queue(sche_queue_t *sche_queue, fifo_queue_t *fifo_queue);

int fifo_queue_init(fifo_queue_t *fifo_queue) {
    fifo_queue->item_cnt = 0;
    RETURN_IF_ERROR(pthread_mutex_init(&fifo_queue->fifo_lock, NULL),
                    FIFO_QUEUE_ERROR_INIT_FIFO_LOCK_FAILED);
    fifo_queue->queue = WorkQueue<fifo_queue_item_t>();
    return SUCCESS;
}

int fast_queue_init(fast_queue_t *fast_queue) {
    fast_queue->tellmenow_req_cnt = 0;
    fast_queue->projectidea_req_cnt = 0;
    RETURN_IF_ERROR(pthread_mutex_init(&fast_queue->tellmenow_mutex, NULL),
                    FAST_QUEUE_ERROR_INIT_LOCK_FAILED);
    RETURN_IF_ERROR(pthread_mutex_init(&fast_queue->projectidea_mutex, NULL),
                    FAST_QUEUE_ERROR_INIT_LOCK_FAILED);
    fast_queue->tellmenow_queue = WorkQueue<Request_msg>();
    fast_queue->projectidea_queue = WorkQueue<Request_msg>();
    return SUCCESS;
}

int sche_queue_init(sche_queue_t *sche_queue) {
    sche_queue->item_cnt = 0;
    sche_queue->if_being_updated = 0;
    RETURN_IF_ERROR(pthread_mutex_init(&sche_queue->updated_lock, NULL),
                    SHCE_QUEUE_ERROR_INIT_SCHE_LOCK_FAILED);
    RETURN_IF_ERROR(pthread_mutex_init(&sche_queue->sche_lock, NULL),
                    SHCE_QUEUE_ERROR_INIT_SCHE_LOCK_FAILED);
    // TODO WorkQueue does not check return value.
    sche_queue->queue = WorkQueue<Request_msg>();
    return SUCCESS;
}

fifo_queue_item_t create_fifo_queue_item(int complexity,
        const Request_msg& req) {
    fifo_queue_item_t new_item;
    new_item.complexity = complexity;
    new_item.req = req;
    return new_item;
}

void fifo_queue_put(fifo_queue_t *fifo_queue,
                    const Request_msg &req,
                    int req_type) {
    int complexity = 0;
    switch (req_type) {
    case WISDOM418:
        complexity = DEFAULT_COMPLEXITY;
        break;
    case COUNTERPRIMES:
        complexity = std::stoi(req.get_arg("n"));
        break;
    default:
        break;
    }
    fifo_queue_item_t fifo_queue_item = create_fifo_queue_item(complexity, req);
    pthread_mutex_lock(&fifo_queue->fifo_lock);
    fifo_queue->queue.put_work(fifo_queue_item);
    fifo_queue->item_cnt++;
    pthread_mutex_unlock(&fifo_queue->fifo_lock);
    DEBUG_PRINT("put in fifo, tag: %d, req: %s\n",
                req.get_tag(), req.get_request_string().c_str());
}

int fifo_queue_get(fifo_queue_t *fifo_queue, fifo_queue_item_t &item) {
    pthread_mutex_lock(&fifo_queue->fifo_lock);
    if (fifo_queue->item_cnt == 0) {
        pthread_mutex_unlock(&fifo_queue->fifo_lock);
        return 0;
    }
    fifo_queue->item_cnt--;
    item = fifo_queue->queue.get_work();
    pthread_mutex_unlock(&fifo_queue->fifo_lock);
    return 1;
}

void fast_queue_put(fast_queue_t *fast_queue,
                    const Request_msg &req,
                    int req_type) {
    switch (req_type) {
    case PROJECTIDEA:
        fast_queue->projectidea_queue.put_work(req);
        pthread_mutex_lock(&fast_queue->projectidea_mutex);
        fast_queue->projectidea_req_cnt++;
        pthread_mutex_unlock(&fast_queue->projectidea_mutex);
        break;
    case TELLMENOW:
        fast_queue->tellmenow_queue.put_work(req);
        pthread_mutex_lock(&fast_queue->tellmenow_mutex);
        fast_queue->tellmenow_req_cnt++;
        pthread_mutex_unlock(&fast_queue->tellmenow_mutex);
        break;
    default:
        break;
    }
}

int fast_queue_get(fast_queue_t *fast_queue,
                   Request_msg& req,
                   int req_type,
                   int &wait_if_zero,
                   pthread_cond_t *cond) {
    switch (req_type) {
    case PROJECTIDEA:
        pthread_mutex_lock(&fast_queue->projectidea_mutex);
        if (fast_queue->projectidea_req_cnt == 0) {
            pthread_mutex_unlock(&fast_queue->projectidea_mutex);
            return 0;
        }
        fast_queue->projectidea_req_cnt--;
        pthread_mutex_unlock(&fast_queue->projectidea_mutex);
        req = fast_queue->projectidea_queue.get_work();
        break;
    case TELLMENOW:
        pthread_mutex_lock(&fast_queue->tellmenow_mutex);
        while (fast_queue->tellmenow_req_cnt == 0) {
            if (wait_if_zero) {
                pthread_cond_wait(cond, &fast_queue->tellmenow_mutex);
                wait_if_zero = 0;
            } else {
                pthread_mutex_unlock(&fast_queue->tellmenow_mutex);
                return 0;
            }
        }
        fast_queue->tellmenow_req_cnt--;
        pthread_mutex_unlock(&fast_queue->tellmenow_mutex);
        req = fast_queue->tellmenow_queue.get_work();
        break;
    default:
        break;
    }
    return 1;
}

int fill_sche_queue(sche_queue_t *sche_queue, fifo_queue_t *fifo_queue) {
    fifo_queue_item_t fifo_array[SCHEDULER_LENGTH];
    int is_taken[SCHEDULER_LENGTH] = {0};
    int min_complexity = 0;
    int min_idx;
    int actual_len = 0;
    for (actual_len = 0; actual_len < SCHEDULER_LENGTH; actual_len++) {
        fifo_queue_item_t item_temp;
        if (fifo_queue_get(fifo_queue, item_temp) == 1) {
            // DEBUG_PRINT("enter fifo_queue_get!!!!!!, should only happen once\n");
            fifo_array[actual_len] = item_temp;
        } else break;
    }
    if (actual_len == 0) return 0;
    min_complexity = fifo_array[0].complexity;
    min_idx = 0;

    // DEBUG_PRINT("line %d in fill sche_queue, actual_len: %d\n",
    //             __LINE__, actual_len);
    int input_len = 0;
    while (input_len != actual_len) {
        for (int i = 0; i < actual_len; i++) {
            if (is_taken[i] == 0) {
                min_idx = i;
                break;
            }
        }
        for (int i = 1; i < actual_len; i++) {
            int complexity_iter = fifo_array[i].complexity;
            if (min_complexity > complexity_iter && is_taken[i] == 0) {
                min_idx = i;
                min_complexity = complexity_iter;
            }
        }
        input_len++;
        sche_queue->queue.put_work(fifo_array[min_idx].req);
        is_taken[min_idx] = 1;
    }
    // DEBUG_PRINT("line %d in fill sche_queue, input_len: %d\n",
    //             __LINE__, input_len);
    pthread_mutex_lock(&sche_queue->updated_lock);
    sche_queue->item_cnt += (input_len - 1);
    pthread_mutex_unlock(&sche_queue->updated_lock);
    return 1;
}

int sche_queue_get(sche_queue_t *sche_queue,
                   Request_msg &req,
                   fifo_queue_t *fifo_queue) {
    pthread_mutex_lock(&sche_queue->sche_lock);
    if (sche_queue->item_cnt == 0) {
        // if (iiii_global == 1) {
        //     DEBUG_PRINT("line %d in sche_queue_get\n", __LINE__);
        // }
        pthread_mutex_lock(&sche_queue->updated_lock);
        if (sche_queue->if_being_updated == 1) {
            pthread_mutex_unlock(&sche_queue->updated_lock);
            pthread_mutex_unlock(&sche_queue->sche_lock);
            return 0;
        } else {
            sche_queue->if_being_updated = 1;
            pthread_mutex_unlock(&sche_queue->updated_lock);
            pthread_mutex_unlock(&sche_queue->sche_lock);
            int ret = fill_sche_queue(sche_queue, fifo_queue);
            pthread_mutex_lock(&sche_queue->updated_lock);
            sche_queue->if_being_updated = 0;
            pthread_mutex_unlock(&sche_queue->updated_lock);
            if (ret == 0) return 0;
        }
    } else {
        // if (iiii_global == 1) {
        //     DEBUG_PRINT("line %d in sche_queue_get,item_cnt: %d \n",
        //                 __LINE__, sche_queue->item_cnt);
        //     DEBUG_PRINT("line %d in sche_queue_get\n", __LINE__);
        // }
        DEBUG_PRINT("Maybe I got into here\n");
        sche_queue->item_cnt--;
        pthread_mutex_unlock(&sche_queue->sche_lock);
    }
    req = sche_queue->queue.get_work();
    // if (iiii_global == 1)
    //     DEBUG_PRINT("line %d in sche_queue_get, should not print this line\n",
    //                 __LINE__);
    return 1;
}
#endif
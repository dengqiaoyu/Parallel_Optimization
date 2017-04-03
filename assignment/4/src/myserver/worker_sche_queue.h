#ifndef _WORKER_H_
#define _WORKER_H_

// #include <vector>
#include <pthread.h>
#include "server/messages.h"
#include "server/worker.h"
#include "tools/work_queue.h"

#define DEFAULT_COMPLEXITY 1000000
#define SCHEDULER_LENGTH 10

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
                    const Request_msg& req,
                    int req_type);
int fifo_queue_get(fifo_queue_t *fifo_queue, fifo_queue_item_t &item);

int fast_queue_init(fast_queue_t *fast_queue);
void fast_queue_put(fast_queue_t *fast_queue,
                    const Request_msg& req,
                    int req_type);
int fast_queue_get(fast_queue_t *fast_queue,
                   Request_msg& req,
                   int req_type,
                   int wait_if_zero,
                   pthread_cond_t *cond);

int sche_queue_init(sche_queue_t *sche_queue);
int sche_queue_get(sche_queue_t *sche_queue,
                   Request_msg& req,
                   fifo_queue_t *fifo_queue);

#endif
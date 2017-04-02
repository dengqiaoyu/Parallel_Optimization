#include "worker_sche_queue.h"
#include "return_error.h"
#include "request_type_def.h"
#include <pthread.h>

// TODO remember to remove it
void pthread_mutex_lock(pthread_mutex_t mutex, void *ptr);
void pthread_mutex_unlock(pthread_mutex_t mutex, void *ptr);

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
                    const Request_msg& req,
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
    pthread_mutex_lock(fifo_queue->fifo_lock, NULL);
    fifo_queue->queue.put_work(fifo_queue_item);
    fifo_queue->item_cnt++;
    pthread_mutex_unlock(fifo_queue->fifo_lock, NULL);
}

int fifo_queue_get(fifo_queue_t *fifo_queue, fifo_queue_item_t &item) {
    pthread_mutex_lock(fifo_queue->fifo_lock, NULL);
    if (fifo_queue->item_cnt == 0) {
        pthread_mutex_unlock(fifo_queue->fifo_lock, NULL);
        return 0;
    }
    fifo_queue->item_cnt--;
    item = fifo_queue->queue.get_work();
    pthread_mutex_unlock(fifo_queue->fifo_lock, NULL);
    return 1;
}

void fast_queue_put(fast_queue_t *fast_queue,
                    int req_type,
                    const Request_msg& req) {
    switch (req_type) {
    case PROJECTIDEA:
        fast_queue->projectidea_queue.put_work(req);
        pthread_mutex_lock(fast_queue->projectidea_mutex, NULL);
        fast_queue->projectidea_req_cnt++;
        pthread_mutex_unlock(fast_queue->projectidea_mutex, NULL);
        break;
    case TELLMENOW:
        fast_queue->tellmenow_queue.put_work(req);
        pthread_mutex_lock(fast_queue->tellmenow_mutex, NULL);
        fast_queue->tellmenow_req_cnt++;
        pthread_mutex_unlock(fast_queue->tellmenow_mutex, NULL);
        break;
    default:
        break;
    }
}

int fast_queue_get(fast_queue_t *fast_queue, int req_type, Request_msg& req) {
    switch (req_type) {
    case PROJECTIDEA:
        pthread_mutex_lock(fast_queue->projectidea_mutex, NULL);
        if (fast_queue->projectidea_req_cnt == 0) {
            pthread_mutex_unlock(fast_queue->projectidea_mutex, NULL);
            return 0;
        }
        fast_queue->projectidea_req_cnt--;
        pthread_mutex_unlock(fast_queue->projectidea_mutex, NULL);
        req = fast_queue->projectidea_queue.get_work();
        break;
    case TELLMENOW:
        pthread_mutex_lock(fast_queue->tellmenow_mutex, NULL);
        if (fast_queue->tellmenow_req_cnt == 0) {
            pthread_mutex_unlock(fast_queue->tellmenow_mutex, NULL);
            return 0;
        }
        fast_queue->tellmenow_req_cnt--;
        pthread_mutex_unlock(fast_queue->tellmenow_mutex, NULL);
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
            fifo_array[actual_len++] = item_temp;
        } else break;
    }
    if (actual_len == 0) return 0;
    min_complexity = fifo_array[0].complexity;
    min_idx = 0;

    int input_len = 0;
    while (input_len != actual_len) {
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
    pthread_mutex_lock(sche_queue->updated_lock, NULL);
    sche_queue->item_cnt += (input_len - 1);
    pthread_mutex_unlock(sche_queue->updated_lock, NULL);
    return 1;
}

int sche_queue_get(sche_queue_t *sche_queue,
                   Request_msg& req,
                   fifo_queue_t *fifo_queue) {
    pthread_mutex_lock(sche_queue->sche_lock, NULL);
    if (sche_queue->item_cnt == 0) {
        pthread_mutex_lock(sche_queue->updated_lock, NULL);
        if (sche_queue->if_being_updated == 1) {
            pthread_mutex_unlock(sche_queue->updated_lock, NULL);
            pthread_mutex_unlock(sche_queue->sche_lock, NULL);
            return 0;
        } else {
            sche_queue->if_being_updated = 1;
            pthread_mutex_unlock(sche_queue->updated_lock, NULL);
            pthread_mutex_unlock(sche_queue->sche_lock, NULL);
            int ret = fill_sche_queue(sche_queue, fifo_queue);
            pthread_mutex_lock(sche_queue->updated_lock, NULL);
            sche_queue->if_being_updated = 0;
            pthread_mutex_unlock(sche_queue->updated_lock, NULL);
            if (ret == 0) return 0;
        }
    } else {
        sche_queue->item_cnt--;
        pthread_mutex_unlock(sche_queue->sche_lock, NULL);
    }
    req = sche_queue->queue.get_work();
    return 1;
}

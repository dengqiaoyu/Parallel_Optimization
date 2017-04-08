/**
 * @file worker.cpp
 * @brief This file contains the implementation for worker
 * @author Qiaoyu Deng(qdeng), Changkai Zhou(zchangka)
 * @bug There is a rare bug, in the situation where thread is going to block on
 *      cond inside fast_queue_get, but before that some requests other than
 *      tellmenow reache queue, but thread now cannot know about it, so it will
 *      still be blocked, until the next request arrives. This situation will
 *      cause unnecessary blocking, which might harm the performance.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <glog/logging.h>
#include <pthread.h>

/* user defined include */
#include "server/messages.h"
#include "server/worker.h"
#include "tools/cycle_timer.h"
#include "worker_sche_queue.h"
#include "request_type_def.h"
#include "return_error.h"

#define MAX_RUNNING_PROJECTIDEA 2  /* the number of thread that runs in a worker */
#define NUM_THREAD_NUM 36 /* the number of threads that are launched during initialization */

int tid[NUM_THREAD_NUM];  /* thread id */

typedef struct wstate {
    int worker_id;              /* the id of worker that is assigned by master */
    fifo_queue_t fifo_queue;    /* fifo queue is used to save CPU intensive request */
    fast_queue_t fast_queue;    /* fast queue is used to queue tellmenow or projectidea that needs fast response */
    /**
     * this one is used when the requests in fifo queue have different workload,
     * the first SCHEDULER_LENGTH of requests in fifo will be reordered according
     * to their computation time(the size of n).
     */
    sche_queue_t sche_queue;
    int request_running_cnt[NUM_TYPES];
    pthread_mutex_t running_cnt_mutex[NUM_TYPES];
    pthread_cond_t waiting_cond;  /* when there is no request, thread block at this cond */
} wstate_t;
wstate_t wstate;

/* internal function */
/**
 * Function that is run by every thread, to get requests and execute.
 * @param  tid_ptr pointer to thread id
 * @return         NULL
 */
void *worker_exec_request_pthread(void *tid_ptr);

/**
 * Put request to their queue including fast queue or fifo queue.
 * @param req_type request type
 * @param req      request received
 */
void enqueue_request(int req_type, const Request_msg & req);

/**
 * Get request from queue.
 * @param  wstate       worker state struct
 * @param  req          request will be put in here.
 * @param  wait_if_zero Indicate whether thread block at cond, when find no new
 *                      request
 * @return              1 for get new request, 0 for no request get.
 */
int get_req(wstate_t *wstate, Request_msg& req, int& wait_if_zero);

/**
 * Let worker working on that request, and send response back to master
 * @param req request to be executed
 */
void work_on_req(const Request_msg& req);

/**
 * @param  wstate   worker state struct
 * @param  req_type the type of request
 * @return          1 for success, 0 for failed
 */
int increase_running_req_cnt(wstate_t *wstate, int req_type);

/**
 * @param  wstate   worker state struct
 * @param  req_type the type of request
 * @return          1 for success, 0 for failed
 */
void decrease_running_req_cnt(wstate_t *wstate, int req_type);

/**
 * When worker is newly create, this function is called and initialize new
 * worker
 * @param params worker id
 */
void worker_node_init(const Request_msg& params) {

    wstate.worker_id = atoi(params.get_arg("worker_id").c_str());
    DLOG(INFO) << "**** Initializing worker: " << params.get_arg("worker_id") << " ****\n";
    int ret = 0;
    ret = fifo_queue_init(&wstate.fifo_queue);
    if (ret != SUCCESS) {
        DLOG(INFO) << "**** Initializing worker fifo failed: " << params.get_arg("worker_id") << " ****\n";
        exit(-1);
    }

    ret = fast_queue_init(&wstate.fast_queue);
    if (ret != SUCCESS) {
        DLOG(INFO) << "**** Initializing worker fast failed: " << params.get_arg("worker_id") << " ****\n";
        exit(-1);
    }

    ret = sche_queue_init(&wstate.sche_queue);
    if (ret != SUCCESS) {
        DLOG(INFO) << "**** Initializing worker sche failed: " << params.get_arg("worker_id") << " ****\n";
        exit(-1);
    }

    for (int i = 0; i < NUM_TYPES; i++) {
        wstate.request_running_cnt[i] = 0;
        ret = pthread_mutex_init(&wstate.running_cnt_mutex[i], NULL);
        if (ret != 0) break;
    }
    if (ret != 0) {
        DLOG(INFO) << "**** Initializing worker mutex failed: " << params.get_arg("work_id") << " ****\n";
        exit(-1);
    }
    ret = pthread_cond_init(&wstate.waiting_cond, NULL);
    if (ret != 0) {
        DLOG(INFO) << "**** Initializing worker cond failed: " << params.get_arg("work_id") << " ****\n";
        exit(-1);
    }

    pthread_t work_thread[NUM_THREAD_NUM];
    /* create working threads */
    for (int i = 0; i < NUM_THREAD_NUM; i++) {
        tid[i] = i;
        pthread_create(&work_thread[i],
                       NULL,
                       &worker_exec_request_pthread,
                       (&tid[i]));
    }
}

/**
 * When worker receives new request, this function is called, and put new
 * request into their queue.
 * @param req newly received request
 */
void worker_handle_request(const Request_msg& req) {

    DLOG(INFO) << "Worker got request: [" << req.get_tag() << ":" << req.get_request_string() << "]\n";

    int req_type = -1;
    if (req.get_arg("cmd").compare("418wisdom") == 0) {
        req_type = WISDOM418;
    } else if (req.get_arg("cmd").compare("projectidea") == 0) {
        req_type = PROJECTIDEA;
    } else if (req.get_arg("cmd").compare("tellmenow") == 0) {
        req_type = TELLMENOW;
    } else if (req.get_arg("cmd").compare("countprimes") == 0) {
        req_type = COUNTERPRIMES;
    }
    enqueue_request(req_type, req);
}

/**
 * Function that is run by every thread, to get requests and execute.
 * @param  tid_ptr pointer to thread id
 * @return         NULL
 */
void *worker_exec_request_pthread(void *tid_ptr) {
    int ret = 1;
    int wait_if_zero = 0;

    Request_msg req;
    int i_test = 0;
    /* Continue getting requests and working on it. */
    while (1) {
        while (1) {
            ret = get_req(&wstate, req, wait_if_zero);
            if (ret) break;
            /**
             * If last time of scanning for new request return nothing,we
             * let wait_if_zero be 1, which will cause thread block on cond, if
             * it cannot find new request in fast queue.
             */
            wait_if_zero = 1;
        }
        work_on_req(req);
    }

    return NULL;
}

/**
 * Put request to their queue including fast queue or fifo queue.
 * @param req_type request type
 * @param req      request received
 */
void enqueue_request(int req_type, const Request_msg & req) {
    switch (req_type) {
    case WISDOM418:
    case COUNTERPRIMES:
        fifo_queue_put(&wstate.fifo_queue, req, req_type);
        break;
    case PROJECTIDEA:
    case TELLMENOW:
        fast_queue_put(&wstate.fast_queue, req, req_type);
        break;
    default:
        break;
    }
    pthread_cond_signal(&wstate.waiting_cond);
}


/**
 * Get request from queue.
 * @param  wstate       worker state struct
 * @param  req          request will be put in here.
 * @param  wait_if_zero Indicate whether thread block at cond, when find no new
 *                      request
 * @return              1 for get new request, 0 for no request get.
 */
int get_req(wstate_t *wstate, Request_msg& req, int& wait_if_zero) {
    int ret = 0;

    ret = 1;
    /* get tellmenow firstly */
    // ret = increase_running_req_cnt(wstate, TELLMENOW);
    if (ret) {
        ret = fast_queue_get(&wstate->fast_queue,
                             req,
                             TELLMENOW,
                             wait_if_zero,
                             &wstate->waiting_cond);
        if (ret) return 1;
        // else decrease_running_req_cnt(wstate, TELLMENOW);
    }

    /* and then get projectidea */
    ret = increase_running_req_cnt(wstate, PROJECTIDEA);
    if (ret) {
        ret = fast_queue_get(&wstate->fast_queue,
                             req,
                             PROJECTIDEA,
                             wait_if_zero,
                             &wstate->waiting_cond);
        if (ret) return 1;
        else decrease_running_req_cnt(wstate, PROJECTIDEA);
    }

    /* get other CPU intensive work */
    // ret = sche_queue_get(&wstate->sche_queue, req, &wstate->fifo_queue);
    fifo_queue_item_t fifo_queue_item;
    ret = fifo_queue_get(&wstate->fifo_queue, fifo_queue_item);
    req = fifo_queue_item.req;
    if (ret) {
        return 1;
    }
    return 0;
}

/**
 * Let worker working on that request, and send response back to master
 * @param req request to be executed
 */
void work_on_req(const Request_msg& req) {
    Response_msg resp(req.get_tag());
    execute_work(req, resp);
    if (req.get_arg("cmd").compare("projectidea") == 0)
        decrease_running_req_cnt(&wstate, PROJECTIDEA);
    worker_send_response(resp);
}

/**
 * @param  wstate   worker state struct
 * @param  req_type the type of request
 * @return          1 for success, 0 for failed
 */
int increase_running_req_cnt(wstate_t *wstate, int req_type) {
    switch (req_type) {
    case WISDOM418:
    case COUNTERPRIMES:
        // pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        // wstate->request_running_cnt[req_type]++;
        // pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        break;
    case PROJECTIDEA:
        pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        if (wstate->request_running_cnt[req_type] >= MAX_RUNNING_PROJECTIDEA) {
            pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
            return 0;
        } else {
            wstate->request_running_cnt[req_type]++;
            pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        }
        break;
    case TELLMENOW:
        // pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        // wstate->request_running_cnt[req_type]++;
        // pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        break;
    }
    return 1;
}

/**
 * @param  wstate   worker state struct
 * @param  req_type the type of request
 * @return          1 for success, 0 for failed
 */
void decrease_running_req_cnt(wstate_t *wstate, int req_type) {
    switch (req_type) {
    case WISDOM418:
    case COUNTERPRIMES:
        // pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        // wstate->request_running_cnt[req_type]++;
        // pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        break;
    case PROJECTIDEA:
        pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        wstate->request_running_cnt[req_type]--;
        pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        break;
    case TELLMENOW:
        // pthread_mutex_lock(&wstate->running_cnt_mutex[req_type]);
        // wstate->request_running_cnt[req_type]++;
        // pthread_mutex_unlock(&wstate->running_cnt_mutex[req_type]);
        break;
    }
}

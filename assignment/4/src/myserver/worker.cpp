
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <glog/logging.h>
#include <pthread.h>

#include "server/messages.h"
#include "server/worker.h"
#include "tools/cycle_timer.h"
#include "worker_sche_queue.h"
#include "request_type_def.h"
#include "return_error.h"

// #define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

#define MAX_RUNNING_TELLMENOW 4
#define MAX_RUNNING_PROJECTIDEA 2
#define NUM_THREAD_NUM 48

int tid[NUM_THREAD_NUM];

// TODO remember to remove it
// void pthread_mutex_lock(pthread_mutex_t *mutex, void *ptr);
// void pthread_mutex_unlock(pthread_mutex_t *mutex, void *ptr);

typedef struct wstate {
    int worker_id;
    fifo_queue_t fifo_queue;
    fast_queue_t fast_queue;
    sche_queue_t sche_queue;
    int request_running_cnt[NUM_TYPES];
    pthread_mutex_t running_cnt_mutex[NUM_TYPES];
    pthread_cond_t waiting_cond;
} wstate_t;
wstate_t wstate;

/* internal function */
void *worker_exec_request_pthread(void *tid_ptr);
void enqueue_request(int req_type, const Request_msg & req);
int get_req(wstate_t *wstate, Request_msg& req, int& wait_if_zero);
void work_on_req(const Request_msg& req);
int increase_running_req_cnt(wstate_t *wstate, int req_type);
void decrease_running_req_cnt(wstate_t *wstate, int req_type);
// int get_tellmenow(wstate_t *wstate, Request_msg& req, int& wait_if_zero);
// int get_projectidea(wstate_t *wstate, Request_msg& req);

// Generate a valid 'countprimes' request dictionary from integer 'n'
// static void create_computeprimes_req(Request_msg& req, int n) {
//     std::ostringstream oss;
//     oss << n;
//     req.set_arg("cmd", "countprimes");
//     req.set_arg("n", oss.str());
// }

// // Implements logic required by compareprimes command via multiple
// // calls to execute_work.  This function fills in the appropriate
// // response.
// static void execute_compareprimes(const Request_msg& req, Response_msg& resp) {

//     int params[4];
//     int counts[4];

//     // grab the four arguments defining the two ranges
//     params[0] = atoi(req.get_arg("n1").c_str());
//     params[1] = atoi(req.get_arg("n2").c_str());
//     params[2] = atoi(req.get_arg("n3").c_str());
//     params[3] = atoi(req.get_arg("n4").c_str());

//     for (int i = 0; i < 4; i++) {
//         Request_msg dummy_req(0);
//         Response_msg dummy_resp(0);
//         create_computeprimes_req(dummy_req, params[i]);
//         execute_work(dummy_req, dummy_resp);
//         counts[i] = atoi(dummy_resp.get_response().c_str());
//     }

//     if (counts[1] - counts[0] > counts[3] - counts[2])
//         resp.set_response("There are more primes in first range.");
//     else
//         resp.set_response("There are more primes in second range.");
// }


void worker_node_init(const Request_msg& params) {

    // This is your chance to initialize your worker.  For example, you
    // might initialize a few data structures, or maybe even spawn a few
    // pthreads here.  Remember, when running on Amazon servers, worker
    // processes will run on an instance with a dual-core CPU.

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

    for (int i = 0; i < NUM_THREAD_NUM; i++) {
        tid[i] = i;
        pthread_create(&work_thread[i],
                       NULL,
                       &worker_exec_request_pthread,
                       (&tid[i]));
    }
}

void worker_handle_request(const Request_msg& req) {


    // Make the tag of the reponse match the tag of the request.  This
    // is a way for your master to match worker responses to requests.
    // Response_msg resp(req.get_tag());
    DLOG(INFO) << "Worker got request: [" << req.get_tag() << ":" << req.get_request_string() << "]\n";

    DEBUG_PRINT("!!!!!!!!!%d worker got request, request tag: %d, req: %s\n",
                wstate.worker_id,
                req.get_tag(),
                req.get_request_string().c_str());
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
    // // Output debugging help to the logs (in a single worker node
    // // configuration, this would be in the log logs/worker.INFO)


    // double startTime = CycleTimer::currentSeconds();

    // if (req.get_arg("cmd").compare("compareprimes") == 0) {

    //     // The compareprimes command needs to be special cased since it is
    //     // built on four calls to execute_execute work.  All other
    //     // requests from the client are one-to-one with calls to
    //     // execute_work.

    //     execute_compareprimes(req, resp);

    // } else {

    //     // actually perform the work.  The response string is filled in by
    //     // 'execute_work'
    //     execute_work(req, resp);

    // }

    // double dt = CycleTimer::currentSeconds() - startTime;
    // DLOG(INFO) << "Worker completed work in " << (1000.f * dt) << " ms (" << req.get_tag()  << ")\n";

    // // send a response string to the master
    // worker_send_response(resp);
}

void *worker_exec_request_pthread(void *tid_ptr) {
    int tid = *(int *)tid_ptr;
    // printf("Hi, I am thread %d\n", tid);
    int ret = 1;
    int wait_if_zero = 0;
    // exit(-1);
    Request_msg req;
    int i_test = 0;
    while (1) {
        while (1) {
            ret = get_req(&wstate, req, wait_if_zero);
            // DEBUG_PRINT("originally, i: %d should never be greater than 2\n",
            //             i);
            if (ret) {
                break;
            }
            wait_if_zero = 1;
        }
        DEBUG_PRINT("worker %d thread %d, Going to work on request %d, arg: %s\n",
                    wstate.worker_id,
                    tid,
                    req.get_tag(),
                    req.get_request_string().c_str());
        work_on_req(req);
    }

    return NULL;
}

int get_req(wstate_t *wstate, Request_msg& req, int& wait_if_zero) {
    int ret = 0;
    // ret = increase_running_req_cnt(wstate, TELLMENOW);
    ret = 1;
    if (ret) {
        ret = fast_queue_get(&wstate->fast_queue,
                             req,
                             TELLMENOW,
                             wait_if_zero,
                             &wstate->waiting_cond);
        if (ret) return 1;
        // else decrease_running_req_cnt(wstate, TELLMENOW);
    }

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

    ret = sche_queue_get(&wstate->sche_queue, req, &wstate->fifo_queue);
    if (ret) {
        return 1;
    }
    return 0;
}

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

void work_on_req(const Request_msg& req) {
    Response_msg resp(req.get_tag());
    execute_work(req, resp);
    DEBUG_PRINT("worker %d is going to send resp: %s for req %d to master\n",
                wstate.worker_id, resp.get_response().c_str(), req.get_tag());
    if (req.get_arg("cmd").compare("projectidea") == 0)
        decrease_running_req_cnt(&wstate, PROJECTIDEA);
    worker_send_response(resp);
}

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


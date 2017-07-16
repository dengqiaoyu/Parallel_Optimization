/**
 * @failed master.h
 * @brief This file contains function declaretion for master.c and some const
 *        define
 * @author Qiaoyu Deng(qdeng), Changkai Zhou(zchangka)
 * @bug No known bugs
 */
#ifndef _MASTER_H_
#define _MASTER_H_

#include "server/messages.h"
#include "server/master.h"
#include "server/worker.h"

/* load balancer parameter */
#define MAX_WORKERS 8
#define COMPPRI_NUM 4              /* the number of works in compareprimes */
#define MAX_CACHE_SIZE 100000
#define MAX_RUNNING_PROJECTIDEA 2  /* max number of projectidea that can run in one worker */
#define SCALEOUT_THRESHOLD 26      /* the number of cpu intensive works that perform scale out */
#define SCALEIN_THRESHOLD 15       /* the number of cpu intensive works that perform scale in */
#define NUM_THREAD_NUM 36          /* the number of thread that runs in a worker */
#define NUM_CONTEXT 26             /* magic parameter :) */
#define MIN_TIME_BEFORE_GET_KILLED 1 /* waiting time before a worker got killed */
#define MIN_TIME_BEFORE_NEXT_WORKER 0 /* wating time before create another worker */
#define INITIAL_WORKER_NUM 1          /* initial number of workers */

/* break compareprimes into four counrtprimes works */
typedef struct comppri_item {
    int params[COMPPRI_NUM];
    int counts[COMPPRI_NUM];
} comppri_item_t;

typedef struct client_request_item {
    Client_handle client_handle;
    Request_msg client_req;
    int counter_primes_n;
    int request_type;
    int worker_idx;
    int idx_if_compppri;  /* the index of compareprimes if this counterprimes work comes from compareprimes */
} client_request_item_t;

typedef struct my_worker_info {
    Worker_handle worker;
    int num_request_each_type[NUM_TYPES];
    int sum_primes_countprimes;             /* sum of counterprimes n, used for load balancer */
    int time_to_be_killed;
} my_worker_info_t;

/* function declare */
/**
 * Get the index of next worker that will be assigned request by load balancer.
 * @param  request_type the type of current request
 * @return              the index of worker
 */
int get_next_worker_idx(int request_type);

/**
 * Get the index of next worker that will be assigned countprimes request by load balancer.
 * @param  n paramter that needs to compute primes.
 * @return   the index of worker
 */
int get_next_worker_idx_cntpri(int n);

/**
 * Handler compareprimes request by break it into four countprimes request.
 * @param client_handle
 * @param client_req    client request
 */
void handle_comppri_req(Client_handle &client_handle,
                        const Request_msg &client_req);

/**
 * Create new countprimes request
 * @param req request that will be created
 * @param n   parameter that will be entered into request
 */
static void create_comppri_req(Request_msg& req, int n);

/**
 * Handle compareprimes request by groupign all the result from countprimes
 * @param worker_handle
 * @param resp                response from worker
 * @param client_request_item
 */
void handle_comppri_resp(Worker_handle worker_handle,
                         const Response_msg& resp,
                         client_request_item_t &client_request_item);

/**
 * Update time var for scale out and scale in.
 */
void update_time();

/**
 * Check whether the condition for scale out and scale in is meeted.
 * @return 1 for scale out, -1 for scale in, 0 for nothing.
 */
int ck_scale_cond();

/**
 * Perform scale out operation
 * @return 0 as success, -1 as failed
 */
int scale_out();

/**
 * Perform scale in operation.
 * @return 0 as success, never return -1, but reserved for future use.
 */
int scale_in();

/**
 * Check the condition for killing a idle worker and kill them.
 */
void kill_worker();

/**
 * Clear the information structure of worker that is being killed.
 * @param worker_idx the index of worker
 */
void clear_worker_info(int worker_idx);

/**
 * Get the total number of request that a worker is owning.
 * @param  worker_idx the index of worker
 * @return            the total number of request
 */
int get_works_num(int worker_idx);

/**
 * Get the total number of request that a worker is owning.
 * @param  worker_idx the index of worker
 * @return            the total number of request
 */
int get_num_cpu_intensive_per_worker(int worker_id);

/**
 * Get the number of cpu intensive request that a worker owns.
 * @param  worker_idx the index of worker
 * @return            the total number of request
 */
int get_num_projectidea_per_worker(int worker_id);

/**
 * Get a free index that no worker is using.
 * @return free index
 */
int get_free_idx();

/**
 * print the number of work that all worker is running and queuing.
 */
void printf_worker_info();

#endif
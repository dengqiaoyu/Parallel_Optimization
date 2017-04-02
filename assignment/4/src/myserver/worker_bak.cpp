
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
// #include <glog/logging.h>
#include <mutex>
#include <vector>

#include "server/messages.h"
#include "server/worker.h"
#include "tools/cycle_timer.h"
#include "tools/work_queue.h"

#define WISDOM418       0
#define PROJECTIDEA     1
#define TELLMENOW       2
#define COUNTERPRIMES   3
#define COMPAREPRIMES   4
#define NUM_TYPES       5

#define MAX_RUNNING_NUM_PROJECTIDEA 2

typedef struct request_queue {
    WorkQueue<Request_msg> queue[NUM_TYPES];
} request_queue_t;
request_queue_t request_queue;


typedef struct num_request {
    int request_cnt[NUM_TYPES];
    std::mutex request_mutex[NUM_TYPES];
} num_request_t;
num_request_t num_request_running;
num_request_t num_request_pending;

/* function declare */
int work_on_request(int request_type);
void work_and_send(int request_type);
void enqueue_request(int request_type, const Request_msg& req);
int decrease_pending_req_cnt(int request_type);
void increase_running_req_cnt(int request_type);
void decrease_running_req_cnt(int request_type);
int check_running_condition(int request_type);

// Generate a valid 'countprimes' request dictionary from integer 'n'
static void create_computeprimes_req(Request_msg& req, int n) {
    std::ostringstream oss;
    oss << n;
    req.set_arg("cmd", "countprimes");
    req.set_arg("n", oss.str());
}

// Implements logic required by compareprimes command via multiple
// calls to execute_work.  This function fills in the appropriate
// response.
static void execute_compareprimes(const Request_msg& req, Response_msg& resp) {

    int params[4];
    int counts[4];

    // grab the four arguments defining the two ranges
    params[0] = atoi(req.get_arg("n1").c_str());
    params[1] = atoi(req.get_arg("n2").c_str());
    params[2] = atoi(req.get_arg("n3").c_str());
    params[3] = atoi(req.get_arg("n4").c_str());

    for (int i = 0; i < 4; i++) {
        Request_msg dummy_req(0);
        Response_msg dummy_resp(0);
        create_computeprimes_req(dummy_req, params[i]);
        execute_work(dummy_req, dummy_resp);
        counts[i] = atoi(dummy_resp.get_response().c_str());
    }

    if (counts[1] - counts[0] > counts[3] - counts[2])
        resp.set_response("There are more primes in first range.");
    else
        resp.set_response("There are more primes in second range.");
}


void worker_node_init(const Request_msg& params) {
    for (int i = 0; i < NUM_TYPES; i++)
        request_queue.queue[i] = WorkQueue<Request_msg>();
    // This is your chance to initialize your worker.  For example, you
    // might initialize a few data structures, or maybe even spawn a few
    // pthreads here.  Remember, when running on Amazon servers, worker
    // processes will run on an instance with a dual-core CPU.
    for (int i = 0; i < NUM_TYPES; i++) {
        num_request_running.request_cnt[i] = 0;
        num_request_pending.request_cnt[i] = 0;
    }
    DLOG(INFO) << "**** Initializing worker: " << params.get_arg("name") << " ****\n";

}

void worker_handle_request(const Request_msg & req) {

    // Make the tag of the reponse match the tag of the request.  This
    // is a way for your master to match worker responses to requests.
    // Response_msg resp(req.get_tag());

    // Output debugging help to the logs (in a single worker node
    // configuration, this would be in the log logs/worker.INFO)
    DLOG(INFO) << "Worker got request: [" << req.get_tag() << ":" << req.get_request_string() << "]\n";
    int request_type = -1;

    if (req.get_arg("cmd").compare("wisdom418") == 0) {
        request_type = WISDOM418;
    } else if (req.get_arg("cmd").compare("projectidea") == 0) {
        request_type = PROJECTIDEA;
    } else if (req.get_arg("cmd").compare("tellmenow") == 0) {
        request_type = TELLMENOW;
    } else if (req.get_arg("cmd").compare("counterprimes") == 0) {
        request_type = COUNTERPRIMES;
    } else if (req.get_arg("cmd").compare("compareprimes") == 0) {
        request_type = COMPAREPRIMES;
    }

    enqueue_request(request_type, req);

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

void worker_handle_request_pthread(int tid) {
    while (1) {
        // starvasion, maybe always assign only one thread to do TELLMENOW
        while (work_on_request(TELLMENOW)) {}
        while (work_on_request(PROJECTIDEA)) {}

    }
}

int work_on_request(int request_type) {
    if (!check_running_condition(request_type))
        return 0;
    increase_running_req_cnt(request_type);
    work_and_send(request_type);
    return 1;
}

void work_and_send(int request_type) {
    Request_msg req = request_queue.queue[request_type].get_work();
    Response_msg resp(req.get_tag());
    execute_work(req, resp);
    decrease_running_req_cnt(request_type);
    worker_send_response(resp);
}

void enqueue_request(int request_type, const Request_msg & req) {
    num_request_pending.request_mutex[request_type].lock();
    num_request_pending.request_cnt[request_type]++;
    num_request_pending.request_mutex[request_type].unlock();
    request_queue.queue[request_type].put_work(req);
}

int decrease_pending_req_cnt(int request_type) {
    num_request_pending.request_mutex[request_type].lock();
    if (num_request_pending.request_cnt[request_type] == 0) {
        num_request_pending.request_mutex[request_type].unlock();
        return 0;
    }
    num_request_pending.request_cnt[request_type]--;
    num_request_pending.request_mutex[request_type].unlock();
    return 1;
}

void increase_running_req_cnt(int request_type) {
    num_request_running.request_mutex[request_type].lock();
    num_request_running.request_cnt[request_type]++;
    num_request_running.request_mutex[request_type].unlock();
}

void decrease_running_req_cnt(int request_type) {
    num_request_running.request_mutex[request_type].lock();
    num_request_running.request_cnt[request_type]--;
    num_request_running.request_mutex[request_type].unlock();
}

int check_running_condition(int request_type) {
    switch (request_type) {
    case TELLMENOW:
        return decrease_pending_req_cnt(TELLMENOW);
    case PROJECTIDEA:
        num_request_running.request_mutex[PROJECTIDEA].lock();
        if (num_request_running.request_cnt[PROJECTIDEA]
                < MAX_RUNNING_NUM_PROJECTIDEA) {
            num_request_running.request_mutex[PROJECTIDEA].unlock();
            return decrease_pending_req_cnt(PROJECTIDEA);
        } else {
            num_request_running.request_mutex[PROJECTIDEA].unlock();
            return 0;
        }
        break;
    default:
        break;
    }
    return 1;
}
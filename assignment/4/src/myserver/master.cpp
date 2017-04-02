//#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <stdlib.h>

#include "server/messages.h"
#include "server/master.h"
#include "tools/work_queue.h"
#include "request_type_def.h"

#define MAX_WORKERS     256


typedef struct client_request_item {
    Client_handle client_handle;
    Request_msg client_req;
    int counter_primes_n;
    int request_type;
    int worker_idx;
} client_request_item_t;

typedef struct my_worker_info {
    int worker_tag;
    Worker_handle worker;
    int num_request_each_type[NUM_TYPES];
} my_worker_info_t;

static struct Master_state {

    // The mstate struct collects all the master node state into one
    // place.  You do not need to preserve any of the fields below, they
    // exist only to implement the basic functionality of the starter
    // code.

    bool server_ready;
    int max_num_workers;
    int next_request_tag;

    my_worker_info_t my_worker[MAX_WORKERS];
    int num_workers;
    std::map<int, client_request_item_t> response_client_map;
} mstate;

int get_next_worker_idx(int request_type);
int get_next_worker_idx_counterprimes(int n);

void master_node_init(int max_workers, int& tick_period) {

    // WorkQueue<client_request_item_t> queue = WorkQueue();
    // set up tick handler to fire every 5 seconds. (feel free to
    // configure as you please)
    tick_period = 5;
    mstate.max_num_workers = max_workers;
    mstate.next_request_tag = 0;

    for (int i = 0; i < MAX_WORKERS; i++) {
        mstate.my_worker[i].worker_tag = 0;
        mstate.my_worker[i].worker = NULL;
        for (int j = 0; j < NUM_TYPES; j++)
            mstate.my_worker[i].num_request_each_type[j] = 0;
    }
    mstate.num_workers = 0;


    // don't mark the server as ready until the server is ready to go.
    // This is actually when the first worker is up and running, not
    // when 'master_node_init' returnes
    mstate.server_ready = false;

    // fire off a request for a new worker

    std::string name_field = "name";
    std::string name_value = "my worker";

    for (int i = 0; i < max_workers; i++) {
        int tag = random();
        Request_msg req(tag);
        req.set_arg(name_field, name_value + std::to_string(i));
        request_new_worker_node(req);
    }
}

void handle_new_worker_online(Worker_handle worker_handle, int tag) {

    // 'tag' allows you to identify which worker request this response
    // corresponds to.  Since the starter code only sends off one new
    // worker request, we don't use it here.

    int idx = mstate.num_workers++;
    mstate.my_worker[idx].worker_tag = tag;
    mstate.my_worker[idx].worker = worker_handle;
    for (int i = 0; i < NUM_TYPES; i++) {
        mstate.my_worker[idx].num_request_each_type[i] = 0;
    }

    // Now that a worker is booted, let the system know the server is
    // ready to begin handling client requests.  The test harness will
    // now start its timers and start hitting your server with requests.
    if (mstate.num_workers == mstate.max_num_workers) {
        server_init_complete();
        mstate.server_ready = true;
    }
}

void handle_worker_response(Worker_handle worker_handle, const Response_msg& resp) {

    // Master node has received a response from one of its workers.
    // Here we directly return this response to the client.

    DLOG(INFO) << "Master received a response from a worker: [" << resp.get_tag() << ":" << resp.get_response() << "]" << std::endl;

    // send_client_response(mstate.waiting_client, resp);

    // mstate.num_pending_client_requests = 0;
    int request_tag = resp.get_tag();
    client_request_item_t client_request_item =
        mstate.response_client_map[request_tag];
    int worker_idx = client_request_item.worker_idx;
    int request_type = client_request_item.request_type;
    int counter_primes_n = client_request_item.counter_primes_n;
    if (request_type == COUNTERPRIMES) {
        mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] -= \
                counter_primes_n;
    } else mstate.my_worker[worker_idx].num_request_each_type[request_type]--;
    mstate.response_client_map.erase(request_tag);
    send_client_response(client_request_item.client_handle, resp);
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

    DLOG(INFO) << "Received request: " << client_req.get_request_string() << std::endl;

    // You can assume that traces end with this special message.  It
    // exists because it might be useful for debugging to dump
    // information about the entire run here: statistics, etc.
    if (client_req.get_arg("cmd") == "lastrequest") {
        Response_msg resp(0);
        resp.set_response("ack");
        send_client_response(client_handle, resp);
        return;
    }

    int request_tag = mstate.next_request_tag++;
    client_request_item_t client_request_item;
    client_request_item.client_handle = client_handle;
    client_request_item.client_req = client_req;


    int worker_idx = 0;
    if (client_req.get_arg("cmd").compare("wisdom418") == 0) {
        worker_idx = get_next_worker_idx(WISDOM418);
        client_request_item.request_type = WISDOM418;
    } else if (client_req.get_arg("cmd").compare("projectidea") == 0) {
        worker_idx = get_next_worker_idx(PROJECTIDEA);
        client_request_item.request_type = PROJECTIDEA;
    } else if (client_req.get_arg("cmd").compare("tellmenow") == 0) {
        worker_idx = get_next_worker_idx(TELLMENOW);
        client_request_item.request_type = TELLMENOW;
    } else if (client_req.get_arg("cmd").compare("counterprimes") == 0) {
        std::string n_string = client_req.get_arg("n");
        int n = std::stoi(n_string);
        worker_idx = get_next_worker_idx_counterprimes(n);
        client_request_item.request_type = COUNTERPRIMES;
        client_request_item.counter_primes_n = n;
    } else if (client_req.get_arg("cmd").compare("compareprimes") == 0) {
        worker_idx = get_next_worker_idx(COMPAREPRIMES);
        client_request_item.request_type = COMPAREPRIMES;
    } else {
        Response_msg resp(0);
        resp.set_response("Oh no! This type of request is not supported by server");
        send_client_response(client_handle, resp);
        mstate.response_client_map.erase(request_tag);
    }

    client_request_item.worker_idx = worker_idx;
    mstate.response_client_map[request_tag] = client_request_item;
    Request_msg worker_req(request_tag, client_req);
    send_request_to_worker(mstate.my_worker, worker_req);
}

int get_next_worker_idx(int request_type) {
    int worker_idx = 0;
    int min_num_request =
        mstate.my_worker[worker_idx].num_request_each_type[request_type];
    for (int i = 1; i < mstate.num_workers; i++) {
        int new_request_num =
            mstate.my_worker[i].num_request_each_type[request_type];
        if (new_request_num < min_num_request) worker_idx = i;
    }
    mstate.my_worker[worker_idx].num_request_each_type[request_type]++;
    return worker_idx;
}

int get_next_worker_idx_counterprimes(int n) {
    int worker_idx = 0;
    int min_num_request =
        mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES];
    for (int i = 1; i < mstate.num_workers; i++) {
        int new_request_num =
            mstate.my_worker[i].num_request_each_type[COUNTERPRIMES];
        if (new_request_num < min_num_request) worker_idx = i;
    }
    mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] += n;
    return worker_idx;
}

void handle_tick() {

    // TODO: you may wish to take action here.  This method is called at
    // fixed time intervals, according to how you set 'tick_period' in
    // 'master_node_init'.

}


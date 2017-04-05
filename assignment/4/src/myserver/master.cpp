#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <sstream>
#include <stdlib.h>

#include "server/messages.h"
#include "server/master.h"
#include "server/worker.h"
#include "tools/work_queue.h"
#include "request_type_def.h"
#include "lru_cache.h"

#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

#define MAX_WORKERS 8
#define COMPPRI_NUM 4
#define MAX_CACHE_SIZE 100000

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
    int idx_if_compppri;
} client_request_item_t;

typedef struct my_worker_info {
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
    std::map<int, comppri_item_t> comppri_map;
} mstate;

lru_cache<std::string, Response_msg> master_cache(MAX_CACHE_SIZE);

void handle_compareprimes_req(Client_handle &client_handle,
                              const Request_msg &client_req);
int get_next_worker_idx(int request_type);
int get_next_worker_idx_counterprimes(int n);
static void create_computeprimes_req(Request_msg& req, int n);
void handle_comppri_response(Worker_handle worker_handle,
                             const Response_msg& resp,
                             client_request_item_t &client_request_item);

void master_node_init(int max_workers, int& tick_period) {

    // WorkQueue<client_request_item_t> queue = WorkQueue();
    // set up tick handler to fire every 5 seconds. (feel free to
    // configure as you please)
    tick_period = 5;
    mstate.max_num_workers = max_workers;
    mstate.next_request_tag = 0;

    for (int i = 0; i < max_workers; i++) {
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

    std::string name_field = "worker_id";
    for (int i = 0; i < max_workers; i++) {
        Request_msg req(i);
        std::string id = std::to_string(i);
        // printf("worker id %s in master", id);
        req.set_arg(name_field, id);
        request_new_worker_node(req);
    }
}

void handle_new_worker_online(Worker_handle worker_handle, int tag) {

    // 'tag' allows you to identify which worker request this response
    // corresponds to.  Since the starter code only sends off one new
    // worker request, we don't use it here.

    // DEBUG_PRINT("worker %d\n", tag);
    mstate.num_workers++;
    int idx = tag;
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
    if (client_request_item.request_type == COMPAREPRIMES)
        return handle_comppri_response(worker_handle, resp, client_request_item);
    int worker_idx = client_request_item.worker_idx;
    int request_type = client_request_item.request_type;
    int counter_primes_n = client_request_item.counter_primes_n;
    if (request_type == COUNTERPRIMES) {
        mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] -= \
                counter_primes_n;
    } else mstate.my_worker[worker_idx].num_request_each_type[request_type]--;
    mstate.response_client_map.erase(request_tag);
    std::string req_desp = client_request_item.client_req.get_request_string();
    DEBUG_PRINT("resp: %s in handle_worker_response\n",
                resp.get_response().c_str());
    master_cache.put(req_desp, resp);
    send_client_response(client_request_item.client_handle, resp);
}

void handle_comppri_response(Worker_handle worker_handle,
                             const Response_msg& resp,
                             client_request_item_t &client_request_item) {
    int request_tag = resp.get_tag();
    int worker_idx = client_request_item.worker_idx;
    int counter_primes_n = client_request_item.counter_primes_n;
    int main_tag = client_request_item.idx_if_compppri;
    int count = atoi(resp.get_response().c_str());
    mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] -= \
            counter_primes_n;

    comppri_item_t comppri_item;
    comppri_item = mstate.comppri_map[main_tag];
    for (int i = 0; i < COMPPRI_NUM; i++) {
        if (comppri_item.params[i] == counter_primes_n) {
            comppri_item.counts[i] = count;
            break;
        }
    }

    int if_completed = 1;
    for (int i = 0; i < COMPPRI_NUM; i++) {
        if (comppri_item.counts[i] == -1) {
            if_completed = 0;
            break;
        }
    }

    if (if_completed) {
        Response_msg resp_comppri;
        if (comppri_item.counts[1] - comppri_item.counts[0] > \
                comppri_item.counts[3] - comppri_item.counts[2])
            resp_comppri.set_response("There are more primes in first range.");
        else
            resp_comppri.set_response("There are more primes in second range.");
        std::string req_desp =
            client_request_item.client_req.get_request_string();
        master_cache.put(req_desp, resp_comppri);
        send_client_response(client_request_item.client_handle, resp_comppri);
        mstate.comppri_map.erase(main_tag);
    }
    mstate.response_client_map.erase(request_tag);
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

    DLOG(INFO) << "Received request: " << client_req.get_request_string() << std::endl;

    DEBUG_PRINT("%s\n", client_req.get_request_string().c_str());
    // You can assume that traces end with this special message.  It
    // exists because it might be useful for debugging to dump
    // information about the entire run here: statistics, etc.
    Response_msg cached_resp;
    std::string cached_req_desp = client_req.get_request_string();
    if (master_cache.exist(cached_req_desp) == true) {
        cached_resp = master_cache.get(cached_req_desp);
        send_client_response(client_handle, cached_resp);
        return;
    }
    if (client_req.get_arg("cmd") == "lastrequest") {
        Response_msg resp(0);
        resp.set_response("ack");
        send_client_response(client_handle, resp);
        return;
    }

    if (client_req.get_arg("cmd").compare("compareprimes") == 0)
        return handle_compareprimes_req(client_handle, client_req);

    int request_tag = mstate.next_request_tag++;
    client_request_item_t client_request_item;
    client_request_item.client_handle = client_handle;
    client_request_item.client_req = client_req;

    int worker_idx = 0;
    // TODO modified for dubug use.
    if (client_req.get_arg("cmd").compare("418wisdom") == 0) {
        worker_idx = get_next_worker_idx(WISDOM418);
        DEBUG_PRINT("worker_idx: %d from master\n", worker_idx);
        client_request_item.request_type = WISDOM418;
    } else if (client_req.get_arg("cmd").compare("projectidea") == 0) {
        worker_idx = get_next_worker_idx(PROJECTIDEA);
        client_request_item.request_type = PROJECTIDEA;
    } else if (client_req.get_arg("cmd").compare("tellmenow") == 0) {
        worker_idx = get_next_worker_idx(TELLMENOW);
        client_request_item.request_type = TELLMENOW;
    } else if (client_req.get_arg("cmd").compare("counterprimes") == 0) {
        int n = atoi(client_req.get_arg("n").c_str());
        worker_idx = get_next_worker_idx_counterprimes(n);
        client_request_item.request_type = COUNTERPRIMES;
        client_request_item.counter_primes_n = n;
    } else {
        Response_msg resp(0);
        resp.set_response("Oh no! This type of request is not supported by server");
        send_client_response(client_handle, resp);
        mstate.response_client_map.erase(request_tag);
        return;
    }

    client_request_item.worker_idx = worker_idx;
    mstate.response_client_map[request_tag] = client_request_item;
    Request_msg worker_req(request_tag, client_req);
    send_request_to_worker(mstate.my_worker[worker_idx].worker, worker_req);
}

void handle_compareprimes_req(Client_handle &client_handle,
                              const Request_msg &client_req) {
    int main_tag = mstate.next_request_tag++;
    int req_tag[COMPPRI_NUM];
    client_request_item_t clt_req_item[COMPPRI_NUM];
    comppri_item_t comppri_item;
    comppri_item.params[0] = atoi(client_req.get_arg("n1").c_str());
    comppri_item.params[1] = atoi(client_req.get_arg("n2").c_str());
    comppri_item.params[2] = atoi(client_req.get_arg("n3").c_str());
    comppri_item.params[3] = atoi(client_req.get_arg("n4").c_str());

    int if_cached[COMPPRI_NUM];
    int cnt_cached[COMPPRI_NUM];
    for (int i = 0; i < COMPPRI_NUM; i++) {
        Request_msg dummy_req(0);
        Response_msg dummy_resp(0);
        create_computeprimes_req(dummy_req, comppri_item.params[i]);
        std::string req_desp = dummy_req.get_request_string();
        if (master_cache.exist(req_desp) == true) {
            if_cached[i] = 1;
            dummy_resp = master_cache.get(req_desp);
            cnt_cached[i] = atoi(dummy_resp.get_response().c_str());
        }
    }


    for (int i = 0; i < COMPPRI_NUM; i++) {
        if (if_cached[i]) continue;
        req_tag[i] = mstate.next_request_tag++;
        clt_req_item[i].client_handle = client_handle;
        clt_req_item[i].client_req = client_req;
        clt_req_item[i].counter_primes_n = comppri_item.params[i];
        clt_req_item[i].request_type = COMPAREPRIMES;
        clt_req_item[i].worker_idx =
            get_next_worker_idx_counterprimes(comppri_item.params[i]);
        clt_req_item[i].idx_if_compppri = main_tag;
    }

    int if_all_cached = 1;
    for (int i = 0; i < COMPPRI_NUM; i++) {
        if (if_cached[i])
            comppri_item.counts[i] = cnt_cached[i];
        else {
            if_all_cached = 0;
            comppri_item.counts[i] = -1;
        }
    }

    if (if_all_cached) {
        Response_msg resp_comppri;
        if (comppri_item.counts[1] - comppri_item.counts[0] > \
                comppri_item.counts[3] - comppri_item.counts[2])
            resp_comppri.set_response("There are more primes in first range.");
        else
            resp_comppri.set_response("There are more primes in second range.");
        std::string req_desp = client_req.get_request_string();
        master_cache.put(req_desp, resp_comppri);
        send_client_response(client_handle, resp_comppri);
    } else {
        mstate.comppri_map[main_tag] = comppri_item;
        for (int i = 0; i < COMPPRI_NUM; i++) {
            if (if_cached[i]) continue;
            Request_msg req_created(0);
            create_computeprimes_req(req_created, comppri_item.params[i]);
            Request_msg worker_req(req_tag[i], req_created);
            mstate.response_client_map[req_tag[i]] = clt_req_item[i];
            send_request_to_worker(mstate.my_worker[clt_req_item[i].worker_idx].worker,
                                   worker_req);
        }
    }
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
        if (new_request_num < min_num_request) {
            min_num_request = new_request_num;
            worker_idx = i;
        }
    }
    mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] += n;
    return worker_idx;
}

static void create_computeprimes_req(Request_msg& req, int n) {
    std::ostringstream oss;
    oss << n;
    req.set_arg("cmd", "countprimes");
    req.set_arg("n", oss.str());
}

void handle_tick() {

    // TODO: you may wish to take action here.  This method is called at
    // fixed time intervals, according to how you set 'tick_period' in
    // 'master_node_init'.

}

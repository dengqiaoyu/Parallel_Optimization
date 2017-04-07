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

// #define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT printf
#else
#define DEBUG_PRINT(...)
#endif

#define LOG_P
#ifdef LOG_P
#define LOG_PRINT printf
#else
#define LOG_PRINT(...)
#endif

// #define DEBUG2
#ifdef DEBUG2
#define DEBUG2_PRINT printf
#else
#define DEBUG2_PRINT(...)
#endif

#define MAX_WORKERS 8
#define COMPPRI_NUM 4
#define MAX_CACHE_SIZE 100000
#define MAX_RUNNING_PROJECTIDEA 2
#define SCALEOUT_THRESHOLD 26
#define SCALEIN_THRESHOLD 15
#define NUM_THREAD_NUM 36
#define NUM_CONTEXT 26
#define MIN_TIME_BEFORE_GET_KILLED 1
#define MIN_TIME_BEFORE_NEXT_WORKER 0
#define INITIAL_WORKER_NUM 1

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
    int sum_primes_countprimes;
    int time_to_be_killed;
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
    int num_workers_run;
    int num_workers_recv;
    int num_workers_plan;
    int idx_array[MAX_WORKERS];

    std::map<int, client_request_item_t> response_client_map;
    std::map<int, comppri_item_t> comppri_map;

    int num_cpu_intensive;
    int num_projectidea;
    int time_since_last_new;
    int if_booted;
    int if_scaling_out;
} mstate;

lru_cache<std::string, Response_msg> master_cache(MAX_CACHE_SIZE);

void handle_compareprimes_req(Client_handle &client_handle,
                              const Request_msg &client_req);
int get_next_worker_idx(int request_type);
int get_next_worker_idx_countprimes(int n);
static void create_computeprimes_req(Request_msg& req, int n);
void handle_comppri_response(Worker_handle worker_handle,
                             const Response_msg& resp,
                             client_request_item_t &client_request_item);
void printf_worker_info();

void update_time();
int ck_scale_cond();
int scale_out();
int scale_in();
void kill_worker();
void clear_worker_info(int worker_idx);
int get_works_num(int worker_idx);
int get_num_cpu_intensive_per_worker(int worker_id);
int get_num_projectidea_per_worker(int worker_id);
int get_free_idx();

void master_node_init(int max_workers, int& tick_period) {

    // WorkQueue<client_request_item_t> queue = WorkQueue();
    // set up tick handler to fire every 5 seconds. (feel free to
    // configure as you please)
    tick_period = 1;
    mstate.max_num_workers = max_workers;
    mstate.next_request_tag = 0;
    mstate.time_since_last_new = 0;
    mstate.if_booted = 0;
    mstate.if_scaling_out = 0;

    for (int i = 0; i < max_workers; i++) {
        mstate.my_worker[i].worker = NULL;
        for (int j = 0; j < NUM_TYPES; j++)
            mstate.my_worker[i].num_request_each_type[j] = 0;
        mstate.my_worker[i].sum_primes_countprimes = 0;
        mstate.my_worker[i].time_to_be_killed = -1;
    }
    mstate.num_workers_plan = INITIAL_WORKER_NUM;
    mstate.num_workers_run = 0;
    mstate.num_workers_recv = 0;


    // don't mark the server as ready until the server is ready to go.
    // This is actually when the first worker is up and running, not
    // when 'master_node_init' returnes
    mstate.server_ready = false;

    // fire off a request for a new worker

    std::string name_field = "worker_id";
    for (int i = 0; i < mstate.num_workers_plan; i++) {
        int worker_idx = i;
        mstate.idx_array[i] = worker_idx;
        Request_msg req(worker_idx);
        std::string id = std::to_string(worker_idx);
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
    mstate.num_workers_run++;
    mstate.num_workers_recv++;
    int idx = tag;
    mstate.my_worker[idx].worker = worker_handle;
    mstate.my_worker[idx].sum_primes_countprimes = 0;
    mstate.my_worker[idx].time_to_be_killed = -1;

    LOG_PRINT("########################\n");
    LOG_PRINT("New node online\n");
    LOG_PRINT("########################\n");
    // Now that a worker is booted, let the system know the server is
    // ready to begin handling client requests.  The test harness will
    // now start its timers and start hitting your server with requests.
    if (mstate.if_booted && mstate.num_workers_run == mstate.num_workers_plan) {
        mstate.if_scaling_out = 0;
    }
    if (!mstate.if_booted && mstate.num_workers_run == mstate.num_workers_plan) {
        server_init_complete();
        mstate.server_ready = true;
        mstate.if_booted = 1;
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
    int request_type = client_request_item.request_type;
    if (request_type == WISDOM418 || request_type == COUNTERPRIMES) {
        mstate.num_cpu_intensive--;
    } else if (request_type == COMPAREPRIMES) {
        mstate.num_cpu_intensive--;
        return handle_comppri_response(worker_handle,
                                       resp, client_request_item);
    } else if (request_type == PROJECTIDEA) {
        mstate.num_projectidea--;
    }
    int worker_idx = client_request_item.worker_idx;
    int counter_primes_n = client_request_item.counter_primes_n;
    mstate.my_worker[worker_idx].num_request_each_type[request_type]--;
    if (request_type == COUNTERPRIMES)
        mstate.my_worker[worker_idx].sum_primes_countprimes -= counter_primes_n;
    mstate.response_client_map.erase(request_tag);
    std::string req_desp = client_request_item.client_req.get_request_string();
    master_cache.put(req_desp, resp);
    DEBUG_PRINT("resp: %s in handle_worker_response is put into cache\n",
                req_desp.c_str());
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
    mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES] -= 1;
    mstate.my_worker[worker_idx].sum_primes_countprimes -= counter_primes_n;

    comppri_item_t comppri_item;
    comppri_item = mstate.comppri_map[main_tag];
    for (int i = 0; i < COMPPRI_NUM; i++) {
        if (comppri_item.params[i] == counter_primes_n) {
            comppri_item.counts[i] = count;
            break;
        }
    }
    mstate.comppri_map[main_tag] = comppri_item;

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
        DEBUG_PRINT("cache hit, request: %s\n",
                    client_req.get_request_string().c_str());
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
        mstate.num_cpu_intensive++;
        worker_idx = get_next_worker_idx(WISDOM418);
        DEBUG_PRINT("worker_idx: %d from master\n", worker_idx);
        client_request_item.request_type = WISDOM418;
    } else if (client_req.get_arg("cmd").compare("projectidea") == 0) {
        mstate.num_projectidea++;
        worker_idx = get_next_worker_idx(PROJECTIDEA);
        client_request_item.request_type = PROJECTIDEA;
    } else if (client_req.get_arg("cmd").compare("tellmenow") == 0) {
        worker_idx = get_next_worker_idx(TELLMENOW);
        client_request_item.request_type = TELLMENOW;
    } else if (client_req.get_arg("cmd").compare("countprimes") == 0) {
        mstate.num_cpu_intensive++;
        int n = atoi(client_req.get_arg("n").c_str());
        worker_idx = get_next_worker_idx_countprimes(n);
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

    int if_cached[COMPPRI_NUM] = {0};
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
            get_next_worker_idx_countprimes(comppri_item.params[i]);
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
            mstate.num_cpu_intensive++;
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
    int num_recv = mstate.num_workers_recv;
    int num_run = mstate.num_workers_run;
    int worker_idx = mstate.idx_array[0];
    int min_num_request =
        mstate.my_worker[worker_idx].num_request_each_type[request_type];
    for (int i = 1; i < num_recv; i++) {
        int new_worker_idx = mstate.idx_array[i];
        int new_request_num =
            mstate.my_worker[new_worker_idx].num_request_each_type[request_type];
        if (new_request_num < min_num_request) {
            min_num_request = new_request_num;
            worker_idx = new_worker_idx;
        }
    }
    int proj_worker_idx = worker_idx;
    int proj_min_num_request = min_num_request;
    if (request_type == PROJECTIDEA && min_num_request >= PROJECTIDEA
            && num_recv < num_run) {
        for (int i = num_recv; i < num_run; i++) {
            int new_worker_idx = mstate.idx_array[i];
            int new_request_num =
                mstate.my_worker[new_worker_idx].num_request_each_type[request_type];
            if (new_request_num < proj_min_num_request) {
                proj_min_num_request = proj_worker_idx;
                proj_worker_idx = new_worker_idx;
            }
        }
    }
    if (worker_idx != proj_worker_idx) {
        LOG_PRINT("#################\n");
        LOG_PRINT("Give projectidea to a killing worker\n");
        LOG_PRINT("#################\n");
        worker_idx = proj_worker_idx;
    }
    if (get_num_cpu_intensive_per_worker(worker_idx) >= NUM_CONTEXT - MAX_RUNNING_PROJECTIDEA
            && (request_type == WISDOM418 || COUNTERPRIMES)
            && num_recv < num_run) {
        LOG_PRINT("#################\n");
        LOG_PRINT("RESTART a killing worker in get_idx\n");
        LOG_PRINT("#################\n");
        worker_idx = mstate.idx_array[num_recv];
        mstate.my_worker[worker_idx].time_to_be_killed = -1;
        mstate.num_workers_recv++;
    }

    mstate.my_worker[worker_idx].num_request_each_type[request_type]++;
    return worker_idx;
}

int get_next_worker_idx_countprimes(int n) {
    int worker_idx = mstate.idx_array[0];
    int min_primes_sum =
        mstate.my_worker[worker_idx].sum_primes_countprimes;
    for (int i = 1; i < mstate.num_workers_recv; i++) {
        int new_worker_idx = mstate.idx_array[i];
        int new_request_primes_sum =
            mstate.my_worker[new_worker_idx].sum_primes_countprimes;
        // LOG_PRINT("min_primes_sum: %d, new_request_primes_sum: %d\n",
        //           min_primes_sum, new_request_primes_sum);
        if (new_request_primes_sum < min_primes_sum) {
            min_primes_sum = new_request_primes_sum;
            worker_idx = new_worker_idx;
        }
    }

    if (get_num_cpu_intensive_per_worker(worker_idx) >= NUM_CONTEXT - MAX_RUNNING_PROJECTIDEA
            && mstate.num_workers_recv < mstate.num_workers_run) {
        LOG_PRINT("#################\n");
        LOG_PRINT("RESTART a killing worker in get_idx_countprimes\n");
        LOG_PRINT("#################\n");
        worker_idx = mstate.idx_array[mstate.num_workers_recv];
        mstate.my_worker[worker_idx].time_to_be_killed = -1;
        mstate.num_workers_recv++;
    }

    mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES]++;
    mstate.my_worker[worker_idx].sum_primes_countprimes += n;
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
#ifdef LOG_P
    printf_worker_info();
#endif
    if (mstate.server_ready != true) return;
    // need update time.
    update_time();
    int if_scale = ck_scale_cond();
    if (if_scale == 1) scale_out();
    else if (if_scale == -1) scale_in();
    kill_worker();
}

int ck_scale_cond() {
    int num_workers_recv = mstate.num_workers_recv;
    int num_workers_run = mstate.num_workers_run;
    int ave_cpu_intensive = mstate.num_cpu_intensive / num_workers_recv;
    // int remaining_proj = 0;
    // for (int i = num_workers_recv; i < num_workers_run; i++) {
    //     int worker_idx = mstate.idx_array[i];
    //     remaining_proj +=
    //         mstate.my_worker[worker_idx].num_request_each_type[PROJECTIDEA];
    // }
    // int num_projectidea_now = mstate.num_projectidea - remaining_proj;
    int num_slots_proj = num_workers_run * MAX_RUNNING_PROJECTIDEA;
    int remaining_slots = num_slots_proj - mstate.num_projectidea;
    LOG_PRINT("num_projectidea: %d, num_workers_recv: %d",
              mstate.num_projectidea, num_workers_recv);
    LOG_PRINT("remaining_slots: %d\n", remaining_slots);
    LOG_PRINT("ave_cpu_intensive: %d, \n", ave_cpu_intensive);
    LOG_PRINT("num_workers_recv: %d, \n", num_workers_recv);
    LOG_PRINT("mstate.if_scaling_out: %d\n", mstate.if_scaling_out);

    if (mstate.if_scaling_out) {
        return 0;
    }
    if (num_workers_recv < mstate.max_num_workers
            && mstate.time_since_last_new >= MIN_TIME_BEFORE_NEXT_WORKER) {
        if (ave_cpu_intensive >= SCALEOUT_THRESHOLD
                || (remaining_slots <= 2 && num_workers_run > 1)
                || (remaining_slots <= 1 && num_workers_run == 1)) {
            return 1;
        }
    }
    if (num_workers_recv > 1) {
        if (ave_cpu_intensive < SCALEIN_THRESHOLD && remaining_slots > 3) {
            return -1;
        }
    }
    return 0;
}

int scale_out() {
    int if_scale_back = 0;
    while (mstate.num_workers_recv < mstate.num_workers_run) {
        LOG_PRINT("############################\n");
        LOG_PRINT("Scale back!\n");
        LOG_PRINT("############################\n");
        int worker_idx = mstate.idx_array[mstate.num_workers_recv];
        mstate.my_worker[worker_idx].time_to_be_killed = -1;
        mstate.num_workers_recv++;
        if_scale_back = 1;
        if (if_scale_back && ck_scale_cond() != 1) return 0;
    }

    if (mstate.num_workers_plan < mstate.max_num_workers && ck_scale_cond()) {
        int worker_idx = get_free_idx();
        LOG_PRINT("############################\n");
        LOG_PRINT("worker_idx, worker_idx: %d\n", worker_idx);
        LOG_PRINT("############################\n");
        mstate.idx_array[mstate.num_workers_plan++] = worker_idx;
        mstate.my_worker[worker_idx].worker = NULL;
        for (int i = 0; i < NUM_TYPES; i++)
            mstate.my_worker[worker_idx].num_request_each_type[i] = 0;
        mstate.my_worker[worker_idx].sum_primes_countprimes = 0;
        mstate.my_worker[worker_idx].time_to_be_killed = -1;

        Request_msg req(worker_idx);
        std::string idx_str = std::to_string(worker_idx);
        req.set_arg("worker_id", idx_str);
        request_new_worker_node(req);
        mstate.if_scaling_out = 1;
        LOG_PRINT("############################\n");
        LOG_PRINT("Scale out!\n");
        LOG_PRINT("############################\n");
    } else return -1;
    return 0;
}

int scale_in() {
    if (mstate.num_workers_recv == 1) return -1;
    mstate.num_workers_recv--;
    int worker_idx = mstate.idx_array[mstate.num_workers_recv];
    mstate.my_worker[worker_idx].time_to_be_killed = 0;
    LOG_PRINT("############################\n");
    LOG_PRINT("Scale in!\n");
    LOG_PRINT("############################\n");
    return 0;
}

void update_time() {
    mstate.time_since_last_new++;
    for (int i = 0; i < mstate.num_workers_run; i++) {
        int worker_idx = mstate.idx_array[i];
        if (mstate.my_worker[worker_idx].time_to_be_killed != -1)
            mstate.my_worker[worker_idx].time_to_be_killed++;
    }
}

void kill_worker() {
    if (mstate.if_scaling_out == 1) return;
    for (int i = mstate.num_workers_recv; i < mstate.num_workers_run; i++) {
        int worker_idx = mstate.idx_array[i];
        int num_works = get_works_num(worker_idx);
        LOG_PRINT("############################\n");
        LOG_PRINT("Worker %d, remaining work %d!\n", worker_idx, num_works);
        LOG_PRINT("############################\n");
        if (mstate.my_worker[worker_idx].time_to_be_killed
                >= MIN_TIME_BEFORE_GET_KILLED && num_works == 0) {
            kill_worker_node(mstate.my_worker[worker_idx].worker);
            clear_worker_info(i);
            mstate.num_workers_run--;
            mstate.num_workers_plan--;
            // send close request
            LOG_PRINT("############################\n");
            LOG_PRINT("Kill worker!\n");
            LOG_PRINT("############################\n");
        }
    }
}

void clear_worker_info(int iterator) {
    for (int i = iterator + 1; i < mstate.num_workers_run; i++) {
        mstate.idx_array[i - 1] = mstate.idx_array[i];
    }
}

int get_works_num(int worker_idx) {
    int num_works = 0;
    for (int i = 0; i < NUM_TYPES - 1; i++) {
        num_works += mstate.my_worker[worker_idx].num_request_each_type[i];
    }
    return num_works;
}

int get_num_cpu_intensive_per_worker(int worker_id) {
    int result = mstate.my_worker[worker_id].num_request_each_type[WISDOM418]
                 + mstate.my_worker[worker_id].num_request_each_type[COUNTERPRIMES];
    return result;
}

int get_num_projectidea_per_worker(int worker_id) {
    return mstate.my_worker[worker_id].num_request_each_type[PROJECTIDEA];
}

int get_free_idx() {
    int num_worker_plan = mstate.num_workers_plan;
    if (num_worker_plan == mstate.max_num_workers) return -1;
    int if_occupy[MAX_WORKERS] = {0};
    for (int i = 0; i < num_worker_plan; i++) {
        if_occupy[mstate.idx_array[i]] = 1;
    }

    for (int i = 0; i < MAX_WORKERS; i++) {
        if (if_occupy[i] == 0) return i;
    }
    return -1;
}

void printf_worker_info() {
    LOG_PRINT("\n\n######################################################\n\n");
    for (int i = 0; i < mstate.num_workers_run; i++) {
        int worker_idx = mstate.idx_array[i];
        int num_cpu_intensive = 0;
        int num_works_total = 0;
        for (int j = 0; j < NUM_TYPES - 1; j++) {
            if (j == WISDOM418 || j == COUNTERPRIMES) {
                num_cpu_intensive +=
                    mstate.my_worker[worker_idx].num_request_each_type[j];
            }
            num_works_total +=
                mstate.my_worker[worker_idx].num_request_each_type[j];
        }
        LOG_PRINT("Worker %d, 418wisdom: %d, countprimes: %d, countprimes_sum: %d, cpu_intensive:%d,  projectidea: %d, tellmenow: %d, works_total: %d\n",
                  worker_idx,
                  mstate.my_worker[worker_idx].num_request_each_type[WISDOM418],
                  mstate.my_worker[worker_idx].num_request_each_type[COUNTERPRIMES],
                  mstate.my_worker[worker_idx].sum_primes_countprimes,
                  num_cpu_intensive,
                  mstate.my_worker[worker_idx].num_request_each_type[PROJECTIDEA],
                  mstate.my_worker[worker_idx].num_request_each_type[TELLMENOW],
                  num_works_total
                 );
    }
    LOG_PRINT("\n\n######################################################\n\n");
}


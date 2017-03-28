// Server engine to run Assignment 4 benchmarks

#include <iostream>
#include <fstream>
#include <pthread.h>

#include "mylogging.h"
#include "benchmark.h"
#include "stat.h"
#include "run.h"
#include "tools/cycle_timer.h"
#include "tools/work_queue.h"

// Server parameters.  These can get changed via commandline arguments

// Total number of threads
int nthreads = 32;
// Jobs listed as: { wisdom, countprimes, tellmenow, projectidea }
// How many of each type of job should be run
int job_count[BENCH_END] = { 64, 0, 0, 4 };
// How many of each job can be run at once
int job_limit[BENCH_END] = { 64, 64, 64, 1 };

// Global state: How many jobs of each type need to be run
int job_left[BENCH_END] = { 0 };

// Record to track single job.
// Continues from creation to execution to completion
class Job {
    static int nextjid;  // Use for generating unique ids

 public:
    int benchmark;  // Which benchmark type
    int id;         // Unique ID for job
    int msecs;      // How many milliseconds did it require

    Job(int b) {
	id = nextjid++;
	benchmark = b;
	msecs = 0;
    }
};

int Job::nextjid;  // Initializes to 0

// Maintain two job queues
typedef WorkQueue<Job> JobQueue;

// Jobs waiting to be executed
JobQueue *newJobQueue = NULL;
// Jobs that have completed
JobQueue *doneQueue = NULL;

// Controller.  Generates jobs, collects results, tells threads when to shut down
void controller() {
    int total_jobs = 0;
    // Generate initial jobs
    // Run in reverse priority, putting projectidea jobs at beginning of queue
    for (int b = BENCH_PROJECT; b >= BENCH_WISDOM; b--) {
	job_left[b] = job_count[b];
	total_jobs += job_count[b];
	int njobs = std::min(job_left[b], job_limit[b]);
	for (int j = 0; j < njobs; j++) {
	    Job job(b);
	    newJobQueue->put_work(job);
	}
	job_left[b] -= njobs;
    }

    // Collect completing jobs, while starting remaining ones
    for (int j = 0; j < total_jobs; j++) {
	Job job = doneQueue->get_work();
	int b = job.benchmark;
	report_job(b, job.msecs);
	// See if still have jobs of this type to start
	if (job_left[b] > 0) {
	    job_left[b] --;
	    Job job = Job(b);
	    newJobQueue->put_work(job);
	    LOG(INFO) << "Controller completed and started another job: "
		      << job_names[b] << std::endl;
	} else {
	    LOG(INFO) << "Controller completed job: " << job_names[b] << std::endl;
	}
    }
    // Now tell threads to finish up
    for (int i = 0; i < nthreads; i++) {
	Job job = Job(BENCH_END);
	newJobQueue->put_work(job);
    }
    LOG(INFO) << "Controller finished" << std::endl;
    return;
}

// Run job.  Return true if normal job
bool run_job(Job &job) {
    if (job.benchmark == BENCH_END)
	return false;
    double startTime = CycleTimer::currentSeconds();
    execute_work(job.benchmark);
    double endTime = CycleTimer::currentSeconds();
    job.msecs = (int) (1000.0 * (endTime - startTime));
    return true;
}

// Thread repeatedly gets job and executes it
// Puts result on doneQueue
void *thread_fun(void *arg) {
    int tid = *(int *) arg;
    delete (int *) arg;
    while (true) {
	Job job = newJobQueue->get_work();
	if (!run_job(job))
	    break;
	LOG(INFO) << "Thread " << tid << " completed job "
		  << job.id << ":" << job_names[job.benchmark]
		  << " in " << job.msecs << "ms" << std::endl;
	doneQueue->put_work(job);
    }
    LOG(INFO) << "Thread " << tid << " done" << std::endl;
    return NULL;
}

void start_threads(pthread_t *threads) {
    for (int i = 0; i < nthreads; i++) {
	void *arg = (void *) new int(i);
	pthread_create(&threads[i], NULL, thread_fun, arg);
    }
}

void finish_threads(pthread_t *threads) {
    for (int i = 0; i < nthreads; i++) {
	pthread_join(threads[i], NULL);
    }
}

// Run the server
void run(int nw) {
    nthreads = nw;
    pthread_t threads[nthreads];
    newJobQueue = new JobQueue;
    doneQueue = new JobQueue;
    start_threads(threads);
    double startTime = CycleTimer::currentSeconds();
    controller();
    finish_threads(threads);
    double endTime = CycleTimer::currentSeconds();
    report(nthreads, endTime - startTime);
}


#include "tsynch.h"

/* Reporting.  Print information if at or above specified level of verbosity */
static int verbosity = 0;
static pthread_mutex_t pmutex = PTHREAD_MUTEX_INITIALIZER;

void set_verb_level(int verblevel) {
    verbosity = verblevel;
}

void report(int verblevel, const char *fmt, ...) {
    if (verblevel <= verbosity) {
	pthread_mutex_lock(&pmutex);
	va_list ap;
	va_start(ap, fmt);
	vprintf(fmt, ap);
	printf("\n");
	fflush(stdout);
	pthread_mutex_unlock(&pmutex);
    }
}


/* Performance parameters */
/* Maximum delay (in milliseconds) */
#define MAX_DELAY 10

/* Global data */
static int max_delay = MAX_DELAY;
static bool initialized = false;
pthread_mutex_t  rmutex = PTHREAD_MUTEX_INITIALIZER;

void set_max_delay(int d) {
    max_delay = d;
}

static long int safe_random() {
    pthread_mutex_lock(&rmutex);
    int val = random();
    pthread_mutex_unlock(&rmutex);
    return val;
}

static void safe_srandom(unsigned int seed) {
    pthread_mutex_lock(&rmutex);
    srandom(seed);
    pthread_mutex_unlock(&rmutex);
}

static void initialize() {
    if (initialized)
        return;
    safe_srandom(getpid());
    initialized = true;
}

/* Add more entropy to RNG */
static void entropy() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    int n = t.tv_nsec % 25;
    while (n--)
	safe_random();
}

/* Hard delay.  Won't get interrupted by a signal */
void ms_spin(int msecs) {
    struct timespec t;
    long start_ms;
    long curr_ms;
    clock_gettime(CLOCK_REALTIME, &t);
    start_ms = t.tv_sec * 1000 + t.tv_nsec/1000000;
    curr_ms = start_ms;
    while (curr_ms - start_ms < msecs) {
	clock_gettime(CLOCK_REALTIME, &t);
	curr_ms = t.tv_sec * 1000 + t.tv_nsec/1000000;
    }
}

/* Version of sleep that works at millisecond time scale */
void ms_sleep(int msecs, bool spin) {
    if (msecs == 0)
	return;
    if (spin) {
	ms_spin(msecs);
	return;
    }
	
    struct timespec req;
    long secs = msecs / 1000;
    long nsecs = (msecs - secs * 1000) * 1000000;
    req.tv_sec = secs;
    req.tv_nsec = nsecs;
    if (nanosleep(&req, NULL) < 0) {
        fprintf(stderr, "Couldn't sleep for %d milliseconds\n", msecs);
    }
}

/* Uniform random number in [0.0,1.0] */
static double uniform() {
    initialize();
    return (double) safe_random() / RAND_MAX;
}

int choose_delay() {
    entropy();
    return (int) (uniform() * max_delay);
}

bool env_flag(char *name) {
    return getenv(name) != NULL;
}

bool choose_with_probability(double prob) {
    return uniform() <=  prob;
}

/* Synchronization support.  An overlay of the pthreads library */
Lock::Lock() {
    pthread_mutex_init(&mutex, NULL);
}

void Lock::lock() {
    /* Delay before and after lock acquisition */
    ms_sleep(choose_delay(), true);
    pthread_mutex_lock(&mutex);
    ms_sleep(choose_delay(), true);
}

void Lock::unlock() {
    ms_sleep(choose_delay(), true);
    pthread_mutex_unlock(&mutex);
    ms_sleep(choose_delay(), true);
}

void *thread_fun(void *args) {
    Tester *t = (Tester *) args;
    return t->run_thread();
}

void *Tester::run_thread() {
    // Let the main thread know that thread is ready
    lock.lock();
    done_count++;
    lock.unlock();

    report(5, "Running test thread");
    while (true) {
	lock.lock();
	pthread_cond_wait(&cond, &lock.mutex);
	report(5, "Tester activated with opcount %d", opcount);
	activated_count++;
	lock.unlock();
	int i;
	for (i = 0; i < opcount; i++) {
	    report(5, "Test #%d", i);
	    fun();
	}
	lock.lock();
	done_count++;
	lock.unlock();
	if (opcount == 0)
	    /* This is indication that testing has completed */
	    break;
    }
    return NULL;
}

Tester::Tester(int nt) {
    nthreads = nt;
    threads = new pthread_t[nthreads];
    opcount = 0;
    fun = NULL;
    activated_count = 0;
    done_count = 0;
    pthread_cond_init(&cond, NULL);
    int t;
    for (t = 0; t < nthreads; t++) {
	report(5, "Spawning thread %d", t);
	pthread_create(&threads[t], NULL, thread_fun, (void *) this);
    }
    /* Threads must all get online */
    bool ready = false;
    while (!ready) {
	lock.lock();
	ready = done_count == nthreads;
	lock.unlock();
	ms_sleep(100, true);
    }
}

void Tester::shutdown() {
    /* Tell threads that they should exit */
    runPhase(NULL, 0);
    int t;
    for (t = 0; t < nthreads; t++)
	pthread_join(threads[t], NULL);
}

void Tester::runPhase(run_fun_t f, int ops) {
    fun = f;
    opcount = ops;
    lock.lock();
    activated_count = 0;
    done_count = 0;
    lock.unlock();
    /* Launch all of the threads */
    report(5, "Enabling threads");
    bool activated = false;
    while (!activated) {
	lock.lock();
	pthread_cond_broadcast(&cond);
	activated = activated_count == nthreads;
	lock.unlock();
    }
    bool done  = false;
    while (!done) {
	lock.lock();
	done = done_count == nthreads;
	lock.unlock();
	if (!done)
	    ms_sleep(100, true);
    }
}

ISet::ISet(int lim) {
    limit = lim;
    bitvec = new bool[limit];
    nelements = 0;
}

/* Choose random integer between 0 and lim-1 */
static int ichoose(int lim) {
    double frac = (double) safe_random() / RAND_MAX;
    return (int) (frac * lim);
}

/*
  Choose random value not already in set.
  Insert it and return value.
  Return -1 if set is full
*/
int ISet::addSomeVal() {
    int val;
    lock.lock();
    if (nelements >= limit)
	val = -1;
    else {
	int ord = 1+ichoose(limit - nelements);
	val = 0;
	while (ord > 0) {
	    if (!bitvec[val])
		ord--;
	    if (ord > 0)
		val++;
	}
	bitvec[val] = true;
	nelements++;
    }
    lock.unlock();
    return val;
}

/*
  Choose value in set.
  Delete it and return value.
  Return -1 if set is empty
 */

int ISet::removeSomeVal() {
    int val;
    lock.lock();
    if (nelements == 0)
	val = -1;
    else {
	int ord = 1+ichoose(nelements);
	val = 0;
	while (ord > 0) {
	    if (bitvec[val])
		ord--;
	    if (ord > 0)
		val++;
	}
	bitvec[val] = false;
	nelements--;
    }
    lock.unlock();
    return val;
}

bool ISet::addVal(int val) {
    bool rval = false;
    lock.lock();
    if (!bitvec[val]) {
	bitvec[val] = true;
	nelements++;
	rval = true;
    }
    lock.unlock();
    return rval;
}

bool ISet::removeVal(int val) {
    bool rval = false;
    lock.lock();
    if (bitvec[val]) {
	bitvec[val] = false;
	nelements--;
	rval = true;
    }
    lock.unlock();
    return rval;
}

void ISet::showVals() {
    lock.lock();
    bool first = true;
    for (int val = 0; val < limit; val++) {
	if (bitvec[val]) {
	    if (first)
		printf("[%d", val);
	    else
		printf(" %d", val);
	    first = false;
	}
    }
    printf("]");
    lock.unlock();
}

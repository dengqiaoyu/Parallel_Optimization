/* Enhanced synchronization library.
   Inserts small, random delays into synchronization calls
   to allow more through exercises of different program
   interleavings
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

/* Timer parameters */

/* Set maximum timer delay.  Default = 100 ms */
void set_max_delay(int ms);

/* Choose a random value between 0 and max milliseconds */
int choose_delay();

/* Sleep (or spin) for specified number of milliseconds */
void ms_sleep(int ms, bool spin);

/* Determine if environment value set */
bool env_flag(char *name);

/* Make a boolean choice */
bool choose_with_probability(double prob);

/* Reporting.  Print information if at or above specified level of verbosity */
void set_verb_level(int verblevel);
void report(int verblevel, const char *fmt, ...);

/*** Testing Framework ***/
#define MAXTHREADS 100

// Wrapper around pthread mutex,
// but it also injects random delays
// into locking & unlocking 
struct Lock {
    pthread_mutex_t mutex;
    Lock();  // Constructor initializes the lock
    void lock();
    void unlock();
};

typedef void (*run_fun_t)();

class Tester {
 public:
    int nthreads;
    pthread_t *threads;
    /* Used to coordinate and synchronize the testing thread */
    Lock lock;
    pthread_cond_t cond;
    /* Parameters of current test.  Opcount of 0 indicates time to quit */
    int opcount;
    run_fun_t fun;
    /* Used to determine when all threads are ready */
    int activated_count;
    /* Used to determine when current run is completed */
    int done_count;

    Tester(int nthreads);

    void shutdown();

    // Run single phase of test
    void runPhase(run_fun_t fun, int opcount);

    void *run_thread();
};


/* Thread safe integer set library to enable testing
   of data structures that support insert and delete
*/

class ISet {
 public:
    int limit; /* Max value + 1 */
    int nelements;
    bool *bitvec;
    Lock lock;

    ISet(int limit);

    int addSomeVal();

    int removeSomeVal();

    bool addVal(int val);

    bool removeVal(int val);

    void showVals();
};


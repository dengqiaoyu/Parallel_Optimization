// Crude version of Google logging

// Logging levels
#define DEBUG 0
#define INFO 1
#define WARN 2
#define ERROR 3

extern std::ofstream logfile;

void init_logging(int level, char *progname);
int lognow();
void finish_logging();

#define LOG(level) logfile << "Log (" << lognow() << "):"
//#define LOG(level) std::cout << "Log (" << lognow() << "):"







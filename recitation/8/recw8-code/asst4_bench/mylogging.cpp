#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mylogging.h"

const char *lname[] = { "DEBUG", "INFO", "WARN", "ERROR" };

std::ofstream logfile;

static int beginTime = 0;


char *find_tail(char *pname) {
    int pos = strlen(pname) -1;
    while (pos >= 0 && pname[pos] != '.' && pname[pos] != '/') {
	pos--;
    }
    return &pname[pos+1];
}

void init_logging(int level, char *progname) {
    char buf[50];
    sprintf(buf, "logs/%s.%s", find_tail(progname), lname[level]);
    logfile.open(buf);
    if (!logfile) {
	std::cerr << " Couldn't open log file " << buf << std::endl;
	exit(0);
    }
    beginTime = (int) time(NULL);
}

int lognow() {
    return (int) time(NULL) - beginTime;
}

void finish_logging() {
    logfile.close();
}

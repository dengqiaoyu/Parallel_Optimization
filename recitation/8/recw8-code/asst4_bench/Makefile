CC = g++
CFLAGS = -g -Wall
INC = -I./include
LIB = -lpthread 

all: server

mylogging.o: mylogging.cpp mylogging.h
	$(CC) $(CFLAGS) $(INC) -c mylogging.cpp	

work_engine.o: work_engine.cpp benchmark.h
	$(CC) $(CFLAGS) $(INC) -DBENCH -c work_engine.cpp	

stat.o: stat.cpp benchmark.h stat.h run.h
	$(CC) $(CFLAGS) $(INC) -c stat.cpp

run.o: run.cpp benchmark.h stat.h run.h
	$(CC) $(CFLAGS) $(INC) -c run.cpp

server: main.cpp run.o stat.o work_engine.o mylogging.o
	$(CC) $(CFLAGS) $(INC) -o server main.cpp run.o stat.o work_engine.o mylogging.o $(LIB)

clean:
	rm -f server *.o *~
	rm -rf logs
	mkdir logs


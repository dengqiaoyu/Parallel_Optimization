CC = g++
CFLAGS = -O3 -mfma -fopenmp -Wall 

all: rsort

rsort.o: rsort.cpp rsort.h
	$(CC) $(CFLAGS) -c rsort.cpp

sbench.o: sbench.cpp rsort.h CycleTimer.h
	$(CC) $(CFLAGS) -c sbench.cpp

rsort: main.cpp rsort.o sbench.o
	$(CC) $(CFLAGS) -o rsort main.cpp rsort.o sbench.o

clean:
	rm -f rsort *.o *~ *.s


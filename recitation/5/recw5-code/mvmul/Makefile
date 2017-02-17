CC = g++
#PCFLAGS = -Wall -mavx2 
PCFLAGS = -O3 -mfma -mavx2 -fopenmp -Wall 
CFLAGS = -g $(PCFLAGS)

all: mrun

matrix.o: matrix.cpp matrix.h
	$(CC) $(CFLAGS) -c matrix.cpp

mbench.o: mbench.cpp mbench.h matrix.h
	$(CC) $(CFLAGS) -c mbench.cpp

mvmul.o: mvmul.cpp mbench.h matrix.h
	$(CC) $(CFLAGS) -c mvmul.cpp
	$(CC) $(PCFLAGS) -S mvmul.cpp

mrun: main.cpp matrix.o mbench.o mvmul.o
	$(CC) $(CFLAGS) -o mrun main.cpp matrix.o mbench.o mvmul.o -lm

clean:
	rm -f mrun *.o *~ *.s


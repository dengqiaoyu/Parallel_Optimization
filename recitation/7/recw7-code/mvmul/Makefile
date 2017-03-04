CC = g++
#FPU = -mavx2
FPU = -msse4
CFLAGS = -O3 -mfma $(FPU) -fopenmp -Wall 

all: mrun

matrix.o: matrix.cpp matrix.h
	$(CC) $(CFLAGS) -c matrix.cpp

mbench.o: mbench.cpp mbench.h matrix.h
	$(CC) $(CFLAGS) -c mbench.cpp

mvmul.o: mvmul.cpp mbench.h matrix.h
	$(CC) $(CFLAGS) -c mvmul.cpp

mrun: main.cpp matrix.o mbench.o mvmul.o
	$(CC) $(CFLAGS) -o mrun main.cpp matrix.o mbench.o mvmul.o -lm

clean:
	rm -f mrun *.o *~ *.s


CC=g++
CFLAGS = -g -O0 -Wall

BST = bst.o

all: runbst

bst.o: bst.cpp bst.h tsynch.h
	$(CC) $(INC) $(CFLAGS) -c bst.cpp

driver.o: driver.cpp bst.h driver.h tsynch.h
	$(CC) $(INC) $(CFLAGS) -c driver.cpp

tsynch.o: tsynch.cpp tsynch.h
	$(CC) $(INC) $(CFLAGS) -c tsynch.cpp

runbst: main.cpp $(BST) driver.o tsynch.o
	$(CC) $(CFLAGS) $(INC) -o runbst main.cpp driver.o $(BST) tsynch.o -lpthread -lrt

tar:
	make clean
	(cd ..; tar cf recw11.tar bst)

clean:
	rm -f *.o *~
	rm -f runbst

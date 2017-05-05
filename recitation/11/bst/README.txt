This directory contains code for implementing and testing a
fine-grained binary search tree (BST).  The testing framework embeds
random delays in all of the synchronization operations to cause
different orderings of events to occur each time it is run.

Files:

	tsynch.{cpp,h}: Synchronization and other utility code
	bst.{cpp,h}: Binary search tree code, with various
		     types of synchronization implemented
        driver.{cpp,h}: Testing code
	main.cpp:       Main function
	Makefile
	README.txt

Compiling with the makefile, generates a program runbst with many options.
Run ./runbsd -h to see all of the commandline options

Here are some interesting tests that *should* pass:

./runbst -n 100 -t 1 -P 0.1 -s 20
	 Run single-threaded test of 100 operations, with a set of maximum size 10
	 This will generate a lot of additions and deletions to the BST.
	 The sum operation will occur around 10% of the time (probability 0.1)

./runbst -n 20 -t 5 -s 200 -i -m s
	 Run 5 threads, each performing 20 operations.  Use  only insertion operations.
	 Use simple concurrency

./runbst -n 20 -t 5 -s 200 -i -m s -P 0.1
	 Run 5 threads performing only insertion and sum (probability = 0.1) operations.
	 Use simple concurrency

./runbst -n 20 -t 5 -s 20 -m h
	 Run 5 threads performing only insertion and deletion operations
	 Use hand-over-hand concurrency

./runbst -n 100 -t 5 -s 40 -m h -P 0.1
	 Do a full stress test of the code:  5 threads, 100 operations each, all operation
	 types.  Set cannot exceed 40 elements Use hand-over-hand concurrency.

Here are some interesting tests that may fail	

./runbst -n 100 -t 4 -s 20
	 Run multiple threads with no synchronization

./runbst -n 20 -t 5 -s 200 -i -m s -P 0.1
	 Run code that can't handle concurrent deletions


	

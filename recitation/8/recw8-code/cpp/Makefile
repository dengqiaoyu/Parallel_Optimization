CC=g++
CFLAGS = -O1 -Wall

XFILES = eg1-refparam eg2-string eg2a-string eg2b-string \
	eg3-class eg3a-class eg4-template \
	eg5-deallocate eg5a-deallocate eg5b-deallocate
SFILES = eg1-refparam.s

all: $(XFILES) $(SFILES)

eg1-refparam.s: eg1-refparam.cpp
	$(CC) $(CFLAGS) -S eg1-refparam.cpp

clean:
	rm -f $(XFILES) *~ *.s

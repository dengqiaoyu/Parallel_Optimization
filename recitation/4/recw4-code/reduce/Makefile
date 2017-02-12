EXECUTABLE := reduce

CU_FILES   := reduce.cu

CU_DEPS    := reduce.h

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

EFLAGS =
DEBUG = 1

###########################################################

OBJDIR=objs
CXX=g++ -m64
# This is the normal setting
# CXXFLAGS=-O3 -Wall -g -DDEBUG=$(DEBUG) $(EFLAGS)
# This one is better for debugging
CXXFLAGS=-O0 -Wall -g -DDEBUG=$(DEBUG) $(EFLAGS)

CUDA = 8.0
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
# This is the normal setting
# NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35  -DDEBUG=$(DEBUG) $(EFLAGS)
# Here's a better one for debugging
NVCCFLAGS=-O0 -g -G -m64 --gpu-architecture compute_35 -DDEBUG=$(DEBUG) $(EFLAGS)


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/reduce.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

main.o: reduce.h

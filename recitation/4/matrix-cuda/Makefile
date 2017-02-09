EXECUTABLE := matrix

CU_FILES   := cudaMatrix.cu

CU_DEPS    := matrix.h

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g 
CUDA=cuda-8.0
LDFLAGS=-L/usr/local/depot/$(CUDA)/lib64/ -lcudart
NVCC=/usr/local/depot/$(CUDA)/bin/nvcc
NVCCFLAGS=-O3 -m64 -arch=compute_30

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/matrix.o $(OBJDIR)/cudaMatrix.o


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

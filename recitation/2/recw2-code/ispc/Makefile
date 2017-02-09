
CXX=g++ -m64
CXXFLAGS= -Iobjs/ -O3 -Wall -mavx2 -mfma

ISPC=ispc
SIMD=avx-x2
ISPCFLAGS=-O2 --target=$(SIMD) --arch=x86-64


APP_NAME=sin_ispc
OBJDIR=objs

TASKSYS_CXX=tasksys.cpp
TASKSYS_OBJ=$(OBJDIR)/tasksys.o

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/sin_ispc.o $(TASKSYS_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm -lpthread

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: CycleTimer.h sin.h $(OBJDIR)/sin_ispc.h
$(OBJDIR)/sin.o: sin.h Makefile

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

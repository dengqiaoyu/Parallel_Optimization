
CXX=g++ -m64
CXXFLAGS=-I../common -Iobjs/ -O2 -Wall -mavx2 -mfma

APP_NAME=sin
OBJDIR=objs
CFILE=sin


default: $(APP_NAME) $(APP_NAME).da $(CFILE).s

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME) $(APP_NAME).da

OBJS=$(OBJDIR)/main.o $(OBJDIR)/sin.o 

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm -lpthread

$(APP_NAME).da: $(APP_NAME)
	objdump -d $(APP_NAME) > $(APP_NAME).da

$(CFILE).s: $(CFILE).cpp
	$(CXX) $(CXXFLAGS) -S $(CFILE).cpp

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: CycleTimer.h sin.h
$(OBJDIR)/sin.o: sin.h Makefile


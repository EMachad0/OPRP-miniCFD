
NX=800
NZ=400
FREQ=100
CONFIG=CONFIG_IN_TEST1
SIM_TIME=400

CXX=mpic++
CXXFLAGS=-O3 -std=c++11 -fopenmp
VARIABLES=-D_NX=$(NX) -D_NZ=$(NZ) -D_SIM_TIME=$(SIM_TIME) -D_OUT_FREQ=$(FREQ) -D_IN_CONFIG=$(CONFIG)
RM=rm -f
EXEC=miniCFD

all: clean $(EXEC)

$(EXEC):
	$(CXX) $(CXXFLAGS) $(VARIABLES) $(EXEC).cpp -o $(EXEC).out

clean:
	$(RM) $(EXEC).o $(EXEC).out

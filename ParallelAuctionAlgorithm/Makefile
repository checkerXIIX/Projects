#executable 
HUN_PARALLEL2 = test_hungarian_parallel_man
HUN_SERIAL = test_hungarian_serial
HUN_PARALLEL = test_hungarian_parallel_openmp

AUC_SERIAL = test_auction_serial
AUC_PARALLEL = test_auction_parallel_openmp
AUC_PARALLEL2 = test_auction_parallel_man
#compiler

CPPFLAGS += -O3 -std=c++20 
CPPFLAGS_OPENMP += -O3 -fopenmp -lpthread -std=c++20
CPPFLAGS_PAR += -O3 -pthread -std=c++20 

all: $(HUN_PARALLEL2) $(HUN_SERIAL) $(HUN_PARALLEL) $(AUC_PARALLEL2) $(AUC_SERIAL) $(AUC_PARALLEL)

$(HUN_PARALLEL): $(HUN_PARALLEL).cpp
	$(CXX) $(CPPFLAGS_OPENMP) $^ -o $(HUN_PARALLEL)

$(HUN_SERIAL): $(HUN_SERIAL).cpp
	$(CXX) $(CPPFLAGS) $^ -o $(HUN_SERIAL)

$(HUN_PARALLEL2): $(HUN_PARALLEL2).cpp
	$(CXX) $(CPPFLAGS_PAR) $^ -o $(HUN_PARALLEL2)

$(AUC_PARALLEL2): $(AUC_PARALLEL2).cpp
	$(CXX) $(CPPFLAGS_PAR) $^ -o $(AUC_PARALLEL2)

$(AUC_SERIAL): $(AUC_SERIAL).cpp
	$(CXX) $(CPPFLAGS) $^ -o $(AUC_SERIAL)

$(AUC_PARALLEL): $(AUC_PARALLEL).cpp
	$(CXX) $(CPPFLAGS_OPENMP) $^ -o $(AUC_PARALLEL)

clean:
	rm $(HUN_PARALLEL2) $(HUN_SERIAL) $(HUN_PARALLEL) $(AUC_PARALLEL2) $(AUC_SERIAL) $(AUC_PARALLEL)

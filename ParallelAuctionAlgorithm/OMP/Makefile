AUC_PARALLEL = test_auction_parallel_openmp
CPPFLAGS_OPENMP += -O3 -fopenmp -lpthread -std=c++20

all: $(AUC_PARALLEL)

$(AUC_PARALLEL): $(AUC_PARALLEL).cpp
	$(CXX) $(CPPFLAGS_OPENMP) $^ -o $(AUC_PARALLEL)

clean:
	rm $(AUC_PARALLEL)

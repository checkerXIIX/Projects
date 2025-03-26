AUC_SERIAL = test_auction_serial
CPPFLAGS += -O3 -std=c++20 

all: $(AUC_SERIAL) 

$(AUC_SERIAL): $(AUC_SERIAL).cpp
	$(CXX) $(CPPFLAGS) $^ -o $(AUC_SERIAL)

clean:
	rm $(AUC_SERIAL)

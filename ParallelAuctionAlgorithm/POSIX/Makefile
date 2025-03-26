TARGET = auction

CXX = g++
#compiling flags here

CXXFLAGS = -O3 -std=c++20 -I.

LINKER = $(CXX) -o
#linking flags here
LFLAGS = -I. -pthread

SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = include

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(INCDIR)/*.hpp)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
rm 		 = rm -f

# use clang instead
ifeq ($(CLANG), 1)
	CXX = clang
endif

$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) $@ $(LFLAGS) $(OBJECTS)
	@echo "Linking complete"
	
$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp	
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"
	
.PHONEY: clean
clean:
		@$(rm) $(OBJECTS)
		@$(rm) $(BINDIR)/$(TARGET)
	@echo "Cleanup complete!"

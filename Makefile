# Compilation options
COMPILER ?= g++
VERSION  ?= NAIVE
CXXFLAGS ?= -O0

ifeq ($(VERSION), OPENBLAS)
    LDFLAGS = -lopenblas
else ifeq ($(VERSION), MKL)
    LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
endif

# Input parameters for `make run`
MATRIX_SIZE = 2048

SRC = mat_mult.cc

EXE = mat_mult.exe

.PHONY: all clean run

all: $(EXE)

clean:
	rm -f *.o *.exe

run: $(EXE)
	./$(EXE) $(MATRIX_SIZE)

$(EXE): $(SRC)
	$(COMPILER) -DUSE_$(VERSION) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

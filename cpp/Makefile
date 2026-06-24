CXX      := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math \
            -Iinclude

BUILD    := build
TARGETS  := $(BUILD)/parlay_tests $(BUILD)/parlay_bench

.PHONY: all test bench clean

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/parlay_tests: tests/test_main.cpp include/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD)/parlay_bench: bench/bench_main.cpp include/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

test: $(BUILD)/parlay_tests
	./$(BUILD)/parlay_tests

bench: $(BUILD)/parlay_bench
	./$(BUILD)/parlay_bench

clean:
	rm -rf $(BUILD)

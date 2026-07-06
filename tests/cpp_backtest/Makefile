CXX      := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math \
            -Iinclude

BUILD    := build
TARGETS  := $(BUILD)/parlay_tests $(BUILD)/parlay_bench

.PHONY: all test bench golden clean

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

# Regenerate the cross-language DP golden reference from the Python solver.
# Run this whenever exact_dp.py, stern.py, or bid_model.py change, then
# `make test` to confirm C++ still matches. CI regenerates and diffs this file
# on every run to catch the case where someone forgets.
golden:
	python3 tests/golden/generate_golden.py

clean:
	rm -rf $(BUILD)

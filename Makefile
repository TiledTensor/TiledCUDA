EXAMPLE_DIR := examples
TEST_DIR   	:= tests/python
UNIT_TEST 	?= test_lstm_cell
CPP_UNIT 	:= scripts/unittests/run_all_cpp_test.sh

EXAMPLE 	?= $(EXAMPLE_DIR)/scatter_nd.py
UNIT   		?= $(TEST_DIR)/$(UNIT_TEST).py

WITH_TEST ?= ON

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtiledcuda.so

.PHONY: build example unit_test clean

build:
	@mkdir -p build 
	@cd build && cmake -DWITH_TESTING=$(WITH_TEST) .. && make -j$(proc)

$(DYNAMIC_LIB): build

example: $(DYNAMIC_LIB)
	@python3 $(EXAMPLE)

unit_test: $(DYNAMIC_LIB)
	@python3 $(UNIT)

unit_test_cpp: $(DYNAMIC_LIB)
	@sh $(CPP_UNIT)

clean:
	@rm -rf build

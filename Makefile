EXAMPLE_DIR := examples
TEST_DIR   	:= tests/python
UNIT_TEST 	?= test_scatter_nd.py

EXAMPLE 	?= $(EXAMPLE_DIR)/scatter_nd.py
UNIT   		?= $(TEST_DIR)/$(UNIT_TEST)

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtiledcuda.so

.PHONY: build

build:
	@mkdir -p build 
	@cd build && cmake .. && make

$(DYNAMIC_LIB): build

example: $(DYNAMIC_LIB)
	@python3 $(EXAMPLE)

unit_test: $(DYNAMIC_LIB)
	@python3 $(UNIT)

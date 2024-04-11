EXAMPLE_DIR := examples
EXAMPLE 	?= $(EXAMPLE_DIR)/scatter_nd.py

BUILD_DIR 	:= build
LIB 		:= $(BUILD_DIR)/libtiledcuda.so

.PHONY: build

build:
	@mkdir -p build 
	@cd build && cmake .. && make

$(LIB): build

example: $(LIB)
	@python3 $(EXAMPLE)

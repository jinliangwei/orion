BOSEN_CPP := $(shell find src/bosen -type f -name "*.cpp")
BOSEN_C := $(shell find src/bosen -type f -name "*.c")

BOSEN_HPP := $(shell find include/orion/bosen -type f -name "*.hpp")
BOSEN_HPP += $(shell find include/orion/ -type f -name "*.hpp")
BOSEN_H := $(shell find include/orion/bosen -type f -name "*.h")
BOSEN_H += $(shell find include/orion/ -type f -name "*.h")

BOSEN_LIB_CPP := $(filter-out %_test.cpp %_main.cpp,$(BOSEN_CPP))
BOSEN_LIB_C := $(filter-out %_main.c,$(BOSEN_C))

BOSEN_OBJ := $(BOSEN_CPP:.cpp=.o)
BOSEN_COBJ := $(BOSEN_C:.c=.o)

BOSEN_LIB_OBJ := $(BOSEN_LIB_CPP:.cpp=.o)
BOSEN_LIB_COBJ := $(BOSEN_LIB_C:.c=.o)

BOSEN_MAIN_CPP := $(shell find src/bosen -type f -name "*_main.cpp")
BOSEN_MAIN_C := $(shell find src/bosen -type f -name "*_main.c")

BOSEN_TEST_CPP := $(shell find src/bosen -type f -name "*_test.cpp")
BOSEN_TEST_OBJ := $(BOSEN_TEST_CPP:.cpp=.o)

BOSEN_TEST_EXE := bin/bosen/bosen_test

BOSEN_MAIN_EXE := $(patsubst src/bosen/%_main.cpp,bin/bosen/%,$(BOSEN_MAIN_CPP))
BOSEN_MAIN_EXE += $(patsubst src/bosen/%_main.c,bin/bosen/%,$(BOSEN_MAIN_C))
BOSEN_MAIN_EXE := $(filter-out $(BOSEN_TEST_EXE),$(BOSEN_MAIN_EXE))

BOSEN_PROTO_SRC := $(shell find src/bosen/protobuf -type f -name "*.proto")
BOSEN_PROTO_CPP := $(BOSEN_PROTO_SRC:.proto=.pb.cc)
BOSEN_PROTO_H := $(patsubst src/bosen/protobuf/%.proto,include/orion/bosen/%.pb.h,$(BOSEN_PROTO_SRC))
BOSEN_SRC_PROTO_H := $(patsubst src/bosen/protobuf/%.proto,src/bosen/protobuf/%.pb.h,$(BOSEN_PROTO_SRC))
BOSEN_PROTO_OBJ := $(BOSEN_PROTO_SRC:.proto=.pb.o)
BOSEN_PROTO := $(patsubst src/bosen/protobuf/%.proto,protobuf/%,$(BOSEN_PROTO_SRC))

DRIVER_LIB_CPP := src/bosen/conn.cpp src/bosen/util.cpp src/bosen/byte_buffer.cpp \
		src/bosen/type.cpp \
		src/bosen/julia_evaluator.cpp
DRIVER_LIB_C := src/bosen/driver_c.c src/bosen/constants_c.c

DRIVER_LIB_OBJ := $(DRIVER_LIB_CPP:.cpp=_drlib.o)
DRIVER_LIB_PROTO_OBJ := $(BOSEN_PROTO_OBJ:.pb.o=_drlib.pb.o)
DRIVER_LIB_COBJ := $(DRIVER_LIB_C:.c=_drlib.o)

bosen_exe: $(BOSEN_MAIN_EXE)
bosen_test: $(BOSEN_TEST_EXE)
bosen_proto: $(BOSEN_PROTO_CPP)

bin/bosen: bin
	mkdir -p bin/bosen

$(BOSEN_OBJ): %.o: %.cpp deps $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_H)
	$(CXX) $(CFLAGS) $(SANITIZER_FLAGS) -c $< -o $@

$(BOSEN_COBJ): %.o: %.c deps $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_H)
	$(CXX) $(CFLAGS) $(SANITIZER_FLAGS) -c $< -o $@

$(DRIVER_LIB_OBJ): %_drlib.o: %.cpp deps $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_H)
	$(CXX) $(CFLAGS) -c $< -o $@

$(DRIVER_LIB_PROTO_OBJ): %_drlib.pb.o: %.pb.cc %.pb.h deps $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_H)
	$(CXX) $(CFLAGS) -c $< -o $@

$(DRIVER_LIB_COBJ): %_drlib.o: %.c deps $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_H)
	$(CXX) $(CFLAGS) -c $< -o $@

$(BOSEN_PROTO_OBJ): src/bosen/protobuf/%.o: src/bosen/protobuf/%.cc
	$(CXX) $(CFLAGS) $(SANITIZER_FLAGS) -I$(ORION_HOME)/include/orion/bosen -c $< -o $@

$(BOSEN_TEST_EXE): src/bosen/bosen_test_main.o orion_lib $(BOSEN_TEST_OBJ) bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(BOSEN_TEST_OBJ) $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_MAIN_EXE): bin/%: src/%_main.o orion_lib bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_PROTO): protobuf/%: src/bosen/protobuf/%.proto
	protoc -I=src/bosen/protobuf --cpp_out=src/bosen/protobuf src/bosen/$@.proto

$(BOSEN_PROTO_CPP): src/bosen/protobuf/%.pb.cc : protobuf/%
$(BOSEN_SRC_PROTO_H): src/bosen/protobuf/%.pb.h : protobuf/%
$(BOSEN_PROTO_H): include/orion/bosen/%.pb.h : src/bosen/protobuf/%.pb.h
	cp $< $@

pb_test:
	echo $(BOSEN_PROTO)

bosen_clean:
	rm -rf $(BOSEN_OBJ) $(BOSEN_COBJ) $(DRIVER_LIB_OBJ) $(DRIVER_LIB_COBJ)
	rm -rf $(BOSEN_PROTO_OBJ) $(BOSEN_PROTO_CPP) $(BOSEN_PROTO_H) $(BOSEN_SRC_PROTO_H) $(DRIVER_LIB_PROTO_OBJ)
	rm -rf bin/bosen

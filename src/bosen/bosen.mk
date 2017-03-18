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
BOSEN_PROTO_OBJ := $(BOSEN_PROTO_SRC:.proto=.pb.o)
BOSEN_PROTO := $(patsubst src/bosen/protobuf/%.proto,protobuf/%,$(BOSEN_PROTO_SRC))

bosen_exe: $(BOSEN_MAIN_EXE)
bosen_test: $(BOSEN_TEST_EXE)
bosen_proto: $(BOSEN_PROTO_CPP)

bin/bosen: bin
	mkdir -p bin/bosen

$(BOSEN_OBJ): %.o: %.cpp $(BOSEN_HPP) $(BOSEN_H) $(BOSEN_PROTO_INLCUDE_H)
	$(CXX) -fPIC $(CFLAGS) $(SANITIZER_FLAGS) -c $< -o $@

$(BOSEN_COBJ): %.o: %.c $(BOSEN_HPP) $(BOSEN_H)
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(BOSEN_PROTO_OBJ): src/bosen/protobuf/%.o: src/bosen/protobuf/%.cc include/orion/bosen/%.h
	$(CXX) -fPIC $(CFLAGS) -I$(ORION_HOME)/include/orion/bosen $(SANITIZER_FLAGS) -c $< -o $@

$(BOSEN_TEST_EXE): src/bosen/bosen_test_main.o orion_lib $(BOSEN_TEST_OBJ) bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(BOSEN_TEST_OBJ) $(LDFLAGS) $(ASAN_LIBS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_MAIN_EXE): bin/%: src/%_main.o orion_lib bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(LDFLAGS) $(ASAN_LIBS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_PROTO): protobuf/%: src/bosen/protobuf/%.proto
	protoc -I=src/bosen/protobuf --cpp_out=src/bosen/protobuf src/bosen/$@.proto

$(BOSEN_PROTO_CPP): src/bosen/protobuf/%.pb.cc : protobuf/%
$(BOSEN_PROTO_H): include/orion/bosen/%.pb.h : protobuf/%
	mv src/bosen/protobuf/$*.pb.h $@

pb_test:
	echo $(BOSEN_PROTO)

bosen_clean:
	rm -rf $(BOSEN_OBJ) $(BOSEN_COBJ)
	rm -rf $(BOSEN_PROTO_OBJ) $(BOSEN_PROTO_CPP) $(BOSEN_PROTO_H)
	rm -rf bin/bosen

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

bosen_exe: $(BOSEN_MAIN_EXE)
bosen_test: $(BOSEN_TEST_EXE)

bin/bosen: bin
	mkdir -p bin/bosen

$(BOSEN_OBJ): %.o: %.cpp $(BOSEN_HPP) $(BOSEN_H)
	$(CXX) -fPIC $(CFLAGS) $(SANITIZER_FLAGS) -c $< -o $@

$(BOSEN_COBJ): %.o: %.c $(BOSEN_HPP) $(BOSEN_H)
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(BOSEN_TEST_EXE): src/bosen/bosen_test_main.o orion_lib $(BOSEN_TEST_OBJ) bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(BOSEN_TEST_OBJ) $(LDFLAGS) $(ASAN_LIBS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_MAIN_EXE): bin/%: src/%_main.o orion_lib bin/bosen
	$(CXX) $(SANITIZER_FLAGS) $< $(LDFLAGS) $(ASAN_LIBS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

bosen_clean:
	rm -rf $(BOSEN_OBJ) $(BOSEN_COBJ)
	rm -rf bin/bosen

BOSEN_CPP := $(shell find src/bosen -type f -name "*.cpp")
BOSEN_C := $(shell find src/bosen -type f -name "*.c")
BOSEN_LIB_CPP := $(filter-out %_test.cpp %_main.cpp,$(BOSEN_CPP))
BOSEN_LIB_C := $(filter-out %_test.c %_main.c,$(BOSEN_C))
BOSEN_OBJ := $(BOSEN_CPP:.cpp=.o)
BOSEN_COBJ := $(BOSEN_C:.c=.o)
BOSEN_LIB_OBJ := $(BOSEN_LIB_CPP:.cpp=.o)
BOSEN_LIB_OBJ := $(BOSEN_LIB_C:.c=.o)
BOSEN_MAIN_CPP := $(shell find src/bosen -type f -name "*_main.cpp")
BOSEN_MAIN_C := $(shell find src/bosen -type f -name "*_main.c")
BOSEN_TEST_CPP := $(shell find src/bosen -type f -name "*_test.cpp")
BOSEN_TEST_C := $(shell find src/bosen -type f -name "*_test.c")
BOSEN_MAIN_EXE := $(patsubst src/bosen/%_main.cpp,bin/bosen/%,$(BOSEN_MAIN_CPP))
BOSEN_MAIN_EXE += $(patsubst src/bosen/%_main.c,bin/bosen/%,$(BOSEN_MAIN_C))
BOSEN_TEST_EXE := $(patsubst src/bosen/%_test.cpp,bin/bosen/%_test,$(BOSEN_TEST_CPP))
BOSEN_TEST_EXE += $(patsubst src/bosen/%_test.c,bin/bosen/%_test,$(BOSEN_TEST_C))

bosen_exe: $(BOSEN_MAIN_EXE) $(BOSEN_TEST_EXE)

bin/bosen: bin
	mkdir -p bin/bosen

$(BOSEN_OBJ): %.o: %.cpp
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(BOSEN_COBJ): %.o: %.c
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(BOSEN_MAIN_EXE): bin/%: src/%_main.o $(ORION_LIB) bin/bosen
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_TEST_EXE): bin/%: src/%.o $(ORION_LIB) bin/bosen
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

bosen_clean:
	rm -rf $(BOSEN_OBJ) $(BOSEN_COBJ)
	rm -rf bin/bosen

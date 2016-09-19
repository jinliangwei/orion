BOSEN_CPP := $(shell find src/bosen -type f -name "*.cpp")
BOSEN_LIB_CPP := $(filter-out %_test.cpp %_main.cpp,$(BOSEN_CPP))
BOSEN_OBJ := $(BOSEN_CPP:.cpp=.o)
BOSEN_LIB_OBJ := $(BOSEN_LIB_CPP:.cpp=.o)
BOSEN_MAIN_CPP := $(shell find src/bosen -type f -name "*_main.cpp")
BOSEN_TEST_CPP := $(shell find src/bosen -type f -name "*_test.cpp")
BOSEN_MAIN_EXE := $(patsubst src/bosen/%_main.cpp,bin/bosen/%,$(BOSEN_MAIN_CPP))
BOSEN_TEST_EXE := $(patsubst src/bosen/%_test.cpp,bin/bosen/%_test,$(BOSEN_TEST_CPP))

bosen_lib: $(BOSEN_LIB)
bosen_exe: $(BOSEN_MAIN_EXE) $(BOSEN_TEST_EXE)

bin/bosen: bin
	mkdir -p bin/bosen

$(BOSEN_OBJ): %.o: %.cpp
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(BOSEN_MAIN_EXE): bin/%: src/%_main.o $(ORION_LIB) bin/bosen
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(BOSEN_TEST_EXE): bin/%: src/%.o $(ORION_LIB) bin/bosen
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

bosen_clean:
	rm -rf $(BOSEN_OBJ)
	rm -rf bin/bosen

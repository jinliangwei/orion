SUPPORT_CPP := $(shell find src/support -type f -name "*.cpp")
SUPPORT_LIB_CPP := $(filter-out %_test.cpp %_main.cpp,$(SUPPORT_CPP))
SUPPORT_OBJ := $(SUPPORT_CPP:.cpp=.o)
SUPPORT_LIB_OBJ := $(SUPPORT_LIB_CPP:.cpp=.o)
SUPPORT_MAIN_CPP := $(shell find src/support -type f -name "*_main.cpp")
SUPPORT_TEST_CPP := $(shell find src/support -type f -name "*_test.cpp")
SUPPORT_MAIN_EXE := $(patsubst src/support/%_main.cpp,bin/support/%,$(SUPPORT_MAIN_CPP))
SUPPORT_TEST_EXE := $(patsubst src/support/%_test.cpp,bin/support/%_test,$(SUPPORT_TEST_CPP))

bin/support: bin
	mkdir -p bin/support

support_exe: $(SUPPORT_MAIN_EXE) $(SUPPORT_TEST_EXE)

$(SUPPORT_OBJ): %.o: %.cpp
	$(CXX) -fPIC $(CFLAGS) -c $< -o $@

$(SUPPORT_MAIN_EXE): bin/%: src/%_main.o $(ORION_LIB) bin/support
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

$(SUPPORT_TEST_EXE): bin/%: src/%.o $(ORION_LIB) bin/support
	$(CXX) $< $(LDFLAGS) -l$(ORION_LIB_NAME) $(LIBS) -o $@

support_clean:
	rm -rf $(SUPPORT_OBJ)
	rm -rf bin/support

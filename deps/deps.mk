deps: stx_btree

DEPS_DIR=$(ORION_HOME)/deps
DEPS_INCLUDE_DIR=$(ORION_HOME)/deps/include
DEPS_LIB_DIR=$(ORION_HOME)/deps/lib
DEPS_BIN_DIR=$(ORION_HOME)/deps/bin

stx_btree: $(DEPS_INCLUDE_DIR)/stx

$(DEPS_INCLUDE_DIR)/stx: $(DEPS_DIR)/stx-btree
	cd $(DEPS_DIR)/stx-btree; \
	./configure --prefix=$(DEPS_DIR); \
	make install

deps_clean:
	rm -rf $(DEPS_INCLUDE_DIR)
	rm -rf $(DEPS_LIB_DIR)
	rm -rf $(DEPS_BIN_DIR)

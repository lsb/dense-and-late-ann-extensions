# Native build of the dense_ann SQLite extension.
#
#   make            -> build/dense_ann.so (with pthreads for index building)
#   make test       -> run the Python test suite
#
# For Emscripten / static linking, compile SRC together with sqlite3.c and
# -DSQLITE_CORE (no threads needed; the query path is single-threaded):
#   emcc -O3 -msimd128 -DSQLITE_CORE -I<sqlite dir> $(SRC) sqlite3.c ...
# then call sqlite3_denseann_init(db, 0, 0) or register it with
# sqlite3_auto_extension().

CC      ?= cc
SQLITE_INC ?= /usr/include
CFLAGS  ?= -O3 -march=native -g
CFLAGS  += -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter -fPIC -I$(SQLITE_INC)
THREADS ?= 1
ifeq ($(THREADS),1)
CFLAGS  += -DDENSE_ANN_THREADS -pthread
endif

SRC = dense_ann.c pq.c hnsw.c rawpage.c
HDR = dense_common.h pq.h hnsw.h rawpage.h
OUT = build

all: $(OUT)/dense_ann.so $(OUT)/countvfs.so

$(OUT)/dense_ann.so: $(SRC) $(HDR)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -shared -o $@.tmp $(SRC) -lm && mv $@.tmp $@

$(OUT)/countvfs.so: test/countvfs.c
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -shared -o $@.tmp test/countvfs.c && mv $@.tmp $@

test: all
	python3 test/test_dense_ann.py

clean:
	rm -rf $(OUT)

.PHONY: all test clean

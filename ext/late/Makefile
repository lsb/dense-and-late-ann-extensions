# Native build of the late_plaid SQLite extension (ext/late).
#
#   make            -> build/late.so (entry point sqlite3_late_init), with
#                      pthreads for index building
#   make test       -> run the Python tests
#
# The top-level Makefile (wasm/build.mk) also picks this directory up: it
# builds build/native/ext/late.so and a relocatable WASM object compiled with
# -DSQLITE_CORE (no threads; the query path is single-threaded), and
# registers sqlite3_late_init with sqlite3_auto_extension.

CC      ?= cc
REPO    := ../..
SQLITE_INC ?= $(REPO)/build/sqlite
HV_INC  ?= $(REPO)/wasm/src
CFLAGS  ?= -O3 -march=native -g
CFLAGS  += -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter -fPIC -I$(SQLITE_INC) -I$(HV_INC)
THREADS ?= 1
ifeq ($(THREADS),1)
CFLAGS  += -DLATE_THREADS -pthread
endif

SRC = late_plaid.c late_codec.c late_page.c
HDR = late_common.h late_codec.h late_page.h
OUT = build

all: $(OUT)/late.so

$(OUT)/late.so: $(SRC) $(HDR)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) -lm

test: all
	python3 test/test_late.py

clean:
	rm -rf $(OUT)

.PHONY: all test clean

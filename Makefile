# Makefile for csrc/ — builds the shared library on Linux.
# On Windows, use build.bat instead.
#
# Usage:
#   make          → builds csrc/libbattleship_v2.so
#   make clean    → removes build artefacts

CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -fPIC -std=c11
SRC_DIR := csrc/src
INC_DIR := csrc/include
OUT     := csrc/libbattleship_v2.so

SRCS := $(SRC_DIR)/battleship.c \
        $(SRC_DIR)/placement.c \
        $(SRC_DIR)/env.c

OBJS := $(SRCS:.c=.o)

.PHONY: all clean

all: $(OUT)

$(OUT): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^
	@echo "Built $@"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(OUT)

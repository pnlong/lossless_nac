CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
TARGETS = lpc_predict_helper

all: $(TARGETS)

lpc_predict_helper: lpc_predict_helper.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean 
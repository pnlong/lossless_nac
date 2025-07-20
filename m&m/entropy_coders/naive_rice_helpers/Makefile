CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
TARGETS = naive_rice_encode_helper naive_rice_decode_helper

all: $(TARGETS)

naive_rice_encode_helper: naive_rice_encode_helper.c
	$(CC) $(CFLAGS) -o $@ $<

naive_rice_decode_helper: naive_rice_decode_helper.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean 
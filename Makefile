CC = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/mpic++
CFLAGS = -I. -Ishared-folder/try -Wall -Werror -std=c++17 -O3 -fopenmp

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

all: main

main: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main
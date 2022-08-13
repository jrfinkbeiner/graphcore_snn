CC = g++

all: libcustom_op.so

libcustom_op.so: poplar_code.cpp
	$(CC) poplar_code.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil -lpopnn -lpopops

clean:
	rm libcustom_op.so

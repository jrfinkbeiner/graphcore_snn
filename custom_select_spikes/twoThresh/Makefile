CC = g++

all: libcustom_op.so

custom_codelet.gp: custom_codelet.cpp
	popc -O3 custom_codelet.cpp -o $@

libcustom_op.so: custom_codelet.gp poplar_code.cpp
	$(CC) -O3 poplar_code.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil -lpopnn -lpopops

clean:
	rm libcustom_op.so custom_codelet.gp

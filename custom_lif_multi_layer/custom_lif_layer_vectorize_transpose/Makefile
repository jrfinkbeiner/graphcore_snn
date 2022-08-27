CC = g++ -std=c++17

all: libcustom_op.so

custom_codelet.gp: custom_codelet.cpp
	popc -O3 custom_codelet.cpp -o $@

libcustom_op.so: custom_codelet.gp poplar_code.cpp poplar_functions.cpp
	$(CC) -O3 poplar_code.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil -lpopnn -lpopops -lpoprand

clean:
	rm libcustom_op.so custom_codelet.gp

CUSTOM_CODELET_BASE_PATH := /p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/source

CC = g++ -std=c++17 -DCUSTOM_CODELET_BASE_PATH=$(CUSTOM_CODELET_BASE_PATH) -O3

LIBS = poplar_code.cpp custom_codelet_path.cpp -lpoplar -lpoputil -lpopnn -lpopops -lstdc++fs

all: libcustom_op.so

custom_codelet.gp: custom_codelet.cpp
	popc -O3 custom_codelet.cpp -o $@

libcustom_op.so: custom_codelet.gp custom_op.cpp
	$(CC) custom_op.cpp -shared -fpic -Wl,-soname,$@ -o $@ $(LIBS)

clean:
	rm libcustom_op.so custom_codelet.gp

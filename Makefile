dir_guard=@mkdir -p $(@D)

CUSTOM_CODELET_BASE_PATH := /p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/build/custom_codelets
SO_BASE_PATH := /p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/build/so
SOURCE_BASE_PATH := /p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/source

CC = g++ -std=c++17 -DCUSTOM_CODELET_BASE_PATH=$(CUSTOM_CODELET_BASE_PATH) -O3

LIBS = $(SOURCE_BASE_PATH)/custom_codelet_path.cpp $(SOURCE_BASE_PATH)/string_util.cpp -lpoplar -lpoputil -lpopnn -lpopops -lstdc++fs

BASE_PATH_DNYMATMUL = $(SOURCE_BASE_PATH)/custom_dyn_dense_sparse_matmul/batched/standard
CUSTOM_CODELET_PATH_DNYMATMUL = $(CUSTOM_CODELET_BASE_PATH)/custom_dyn_dense_sparse_matmul/batched/standard
SO_PATH_DNYMATMUL = $(SO_BASE_PATH)/custom_dyn_dense_sparse_matmul/batched/standard

BASE_PATH_SELECTSPIKES = $(SOURCE_BASE_PATH)/custom_select_spikes/twoThresh
CUSTOM_CODELET_PATH_SELECTSPIKES = $(CUSTOM_CODELET_BASE_PATH)/custom_select_spikes/twoThresh
SO_PATH_SELECTSPIKES = $(SO_BASE_PATH)/custom_select_spikes/twoThresh

all: $(SO_PATH_DNYMATMUL)/libcustom_op.so $(SO_PATH_SELECTSPIKES)/libcustom_op.so

# ------------ DYNAMIC MATMUL -------------
$(CUSTOM_CODELET_PATH_DNYMATMUL)/custom_codelet.gp: $(BASE_PATH_DNYMATMUL)/custom_codelet.cpp
	$(dir_guard)
	popc -O3 $(BASE_PATH_DNYMATMUL)/custom_codelet.cpp -o $@

$(SO_PATH_DNYMATMUL)/libcustom_op.so: $(CUSTOM_CODELET_PATH_DNYMATMUL)/custom_codelet.gp $(BASE_PATH_DNYMATMUL)/custom_op.cpp
	$(dir_guard)
	$(CC) $(BASE_PATH_DNYMATMUL)/custom_op.cpp -shared -fpic -Wl,-soname,$@ -o $@ $(BASE_PATH_DNYMATMUL)/poplar_code.cpp $(LIBS)

# ------------ SELECT SPIKES -------------
$(CUSTOM_CODELET_PATH_SELECTSPIKES)/custom_codelet.gp: $(BASE_PATH_SELECTSPIKES)/custom_codelet.cpp
	$(dir_guard)
	popc -O3 $(BASE_PATH_SELECTSPIKES)/custom_codelet.cpp -o $@

$(SO_PATH_SELECTSPIKES)/libcustom_op.so: $(CUSTOM_CODELET_PATH_SELECTSPIKES)/custom_codelet.gp $(BASE_PATH_SELECTSPIKES)/custom_op.cpp
	$(dir_guard)
	$(CC) $(BASE_PATH_SELECTSPIKES)/custom_op.cpp -shared -fpic -Wl,-soname,$@ -o $@ $(BASE_PATH_SELECTSPIKES)/poplar_code.cpp $(LIBS)

clean:
	rm -r $(CUSTOM_CODELET_BASE_PATH)
	rm -r $(SO_BASE_PATH)
# rm libcustom_op.so custom_codelet.gp

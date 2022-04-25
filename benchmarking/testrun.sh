#!/bin/bash

# python3 -c "import glob; print(glob.glob('/p/scratch/chpsadm/finkbeiner1/*'))"
# python3 -c "import glob; print(glob.glob('./*'))"
# python3 -c "import glob; print(glob.glob('/p/scratch/chpsadm/*'))"
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/test"}' python3 -c "import libpvti"

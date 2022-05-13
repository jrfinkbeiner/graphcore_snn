import subprocess
import os

makefiles = subprocess.check_output("find . -type f -name 'Makefile'", shell=True)
print(type(makefiles))

makefiles = makefiles.decode("utf-8") # decode byte array
makefiles = makefiles.split("\n")

for m in makefiles:
    dir = m.replace("Makefile", "")
    print(dir)
    os.system(f"cd {dir} && make all && cd -")
import subprocess

for i in range(2):
    subprocess.run(["sbatch", "performance_jobscript.sh"])
R"""
"""
#
import json
import os
import subprocess


#
for jspath in ("sinfos.json", os.path.join("configs", "sinfos.json")):
    #
    if os.path.isfile(jspath):
        #
        break
if not os.path.isfile(jspath):
    # EXPECT:
    # It is possible slurm information file is not in default locations.
    raise RuntimeError(
        "Fail to find slurm info file (\"sinfos.json\") in any default "
        "locations.",
    )
with open(jspath, "r") as file:
    #
    SINFOS = json.load(file)

#
HOSTNAME = subprocess.check_output(["hostname"]).decode("utf-8").strip()
SERVICES = {
    "cpu": SINFOS["cpus"][SINFOS["services"][HOSTNAME]["cpu"]],
    "cuda": SINFOS["cudas"][SINFOS["services"][HOSTNAME]["cuda"]],
}
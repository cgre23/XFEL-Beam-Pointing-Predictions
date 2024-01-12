#!/bin/bash

# Run DxMAF in its local folder without explicitly having to activate the DxMAF conda environment
# NOTE: Assumes you have a "standard installation", i.e. a dxmaf environment fulfilling the specs from the
# environment.yml file in the DxMAF repository in /home/$USER/.conda/envs/dxmaf.

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")"  # set working dir to location of the dxmaf.sh script (or symlink), so DxMAF output goes there.

exec -a dxmaf /home/${USER}/.conda/envs/dxmaf/bin/python -m dxmaf "$@"  # hand over to python and ensures process name matches for watchdog

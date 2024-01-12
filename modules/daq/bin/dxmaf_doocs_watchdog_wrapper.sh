#!/usr/bin/env bash

# DOOCS Watchdog Wrapper Script for DxMAF
# NOTE: Uses DxMAF environment from lbsync user

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")"  # set working dir to location of the dxmaf.sh script (or symlink), so DxMAF output goes there.

# Default values
CMD="start"
NAME="run" 

# If any of "start,stop,restart" is provided as first argument..
if (( $# >= 1 )) && [[ ${1,,} =~ ^(start|stop|restart)$ ]]; then
    CMD=${1,,}  # .. use it..
    shift  # .. and do not further propagate it.
fi

# Check if a non-default config file is provided ..
while getopts 'c:' flag; do
  case "${flag}" in
    c) NAME="${OPTARG%.*}" ;;  # .. and use that for naming.
  esac
done

case ${CMD} in
  start)
    # default
    ;;
  stop|restart)
    kill -s SIGINT "$(< "${NAME}.PID")"
    sleep 5
    if [ -d "/proc/$(< "${NAME}.PID")" ]; then
      kill -s SIGKILL "$(< "${NAME}.PID")"
    fi
    ;;&
  stop)
    exit 0
    ;;
esac

if [[ "${@: -1}" == "no_pre_start" ]]; then
  set -- "${@:1:$(($#-1))}"  # Remove last argument (i.e. "no_pre_start") https://stackoverflow.com/a/26163980/2175012
fi

echo "$$" > "${NAME}.PID"  # The PID of the current bootstrapper shell will become the PID of DxMAF after exec replaces the process
exec -a ${NAME} /home/lbsync/.conda/envs/dxmaf/bin/python -m dxmaf "$@" &>> "${NAME}.log"  # hand over to python and ensures process name matches for watchdog

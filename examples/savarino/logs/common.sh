# Sourced by every rc9 job: oneAPI env, thread pins, rc9 source over the rc5 venv.
set +u
if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; fi
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RC9=<repo>
PY=<venv>/bin/python
export LINSTABPY_PYTHON=$PY
export PYTHONPATH="$RC9/src:$RC9"
export I_MPI_DEBUG=4 I_MPI_PIN=off
echo "SHA=$(git -C $RC9 rev-parse HEAD) $(git -C $RC9 describe --tags --always) host=$(hostname -s) date=$(date -Is)"
$PY -c "import linstabpy; p=linstabpy.__file__; assert 'linstabpy-rc9' in p, p; print('linstabpy from', p)"

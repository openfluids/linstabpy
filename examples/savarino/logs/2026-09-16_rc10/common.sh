# Sourced by every rc10 job: oneAPI env, thread pins, the released tree.
set +u
if [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; fi
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RC10=<repo>
PY=<venv>/bin/python
export LINSTABPY_PYTHON=$PY
export PYTHONPATH="$RC10/src:$RC10"
export I_MPI_DEBUG=4 I_MPI_PIN=off
echo "TAG=$(git -C $RC10 describe --tags --always) SHA=$(git -C $RC10 rev-parse HEAD) host=$(hostname -s) date=$(date -Is)"
$PY -c "import linstabpy; p=linstabpy.__file__; assert 'rc10runs' in p, p; print('linstabpy from', p)"

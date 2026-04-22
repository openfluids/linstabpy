#!/usr/bin/env bash
# ── linstabpy install diagnostic ────────────────────────────────────────────
# Read-only system check. Reports what is present / missing and prints exact
# package-manager commands to fix any gaps. Nothing is installed or modified.
# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail
trap 'printf "\n[diagnose] interrupted\n"; exit 130' INT TERM

has_cmd() { command -v "$1" >/dev/null 2>&1; }

tool_version() {
    local c="$1"
    local v=""
    v="$("${c}" --version 2>/dev/null | head -n1)" \
        || v="$("${c}" -V 2>/dev/null | head -n1)" \
        || v="$("${c}" -v 2>/dev/null | head -n1)" \
        || v=""
    printf '%s' "${v}"
}

COL_OK='\033[1;32m'
COL_MISS='\033[1;31m'
COL_WARN='\033[1;33m'
COL_INFO='\033[1;34m'
COL_RST='\033[0m'

FAIL_COUNT=0
WARNINGS=()
ACTIONS=()

check_tool() {
    local name="$1"
    local optional="${2:-}"
    if has_cmd "${name}"; then
        local path ver
        path="$(command -v "${name}")"
        ver="$(tool_version "${name}")"
        printf "  %-14s ${COL_OK}OK${COL_RST}   %s" "${name}" "${path}"
        [ -n "${ver}" ] && printf "  (%s)" "${ver}"
        printf '\n'
    else
        if [ "${optional}" = "optional" ]; then
            printf "  %-14s ${COL_WARN}MISS${COL_RST} (optional)\n" "${name}"
        else
            printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "${name}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
}

check_any_tool() {
    local label="$1"
    shift
    local name path ver
    for name in "$@"; do
        if has_cmd "${name}"; then
            path="$(command -v "${name}")"
            ver="$(tool_version "${name}")"
            printf "  %-14s ${COL_OK}OK${COL_RST}   %s" "${label}" "${path}"
            [ "${label}" != "${name}" ] && printf "  (%s)" "${name}"
            [ -n "${ver}" ] && printf "  (%s)" "${ver}"
            printf '\n'
            return 0
        fi
    done
    printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "${label}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
}

intel_mpi_root() {
    if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}" ]; then
        printf '%s\n' "${I_MPI_ROOT%/}"
        return 0
    fi
    return 1
}

intel_mpi_available() {
    intel_mpi_root >/dev/null && return 0
    [ -x /opt/intel/oneapi/setvars.sh ] && return 0
    [ -x /opt/intel/oneapi/mpi/latest/bin/mpicc ] && return 0
    for _cand in /opt/intel/oneapi/mpi/*/bin/mpicc; do
        [ -x "${_cand}" ] && return 0
    done
    return 1
}

resolve_mpi_tool() {
    local root candidate resolved
    root="$(intel_mpi_root || true)"
    if [ -n "${root}" ]; then
        for candidate in "$@"; do
            if [ -x "${root}/bin/${candidate}" ]; then
                printf '%s\n' "${root}/bin/${candidate}"
                return 0
            fi
        done
    fi
    for candidate in "$@"; do
        if has_cmd "${candidate}"; then
            resolved="$(command -v "${candidate}")"
            case "${resolved}" in
                *conda*|*miniconda*|*anaconda*)
                    ;;
                *)
                    printf '%s\n' "${resolved}"
                    return 0
                    ;;
            esac
        fi
    done
    for candidate in "$@"; do
        if [ -x "/usr/bin/${candidate}" ]; then
            printf '%s\n' "/usr/bin/${candidate}"
            return 0
        fi
    done
    for candidate in "$@"; do
        if has_cmd "${candidate}"; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

check_any_mpi_tool() {
    local label="$1"
    shift
    local path ver
    path="$(resolve_mpi_tool "$@" || true)"
    if [ -n "${path}" ]; then
        ver="$(tool_version "${path}")"
        printf "  %-14s ${COL_OK}OK${COL_RST}   %s" "${label}" "${path}"
        [ -n "${ver}" ] && printf "  (%s)" "${ver}"
        printf '\n'
        return 0
    fi
    printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "${label}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
}

mpi_cmd_looks_intel() {
    local cmd="$1"
    local resolved=""
    if [ -x "${cmd}" ]; then
        resolved="${cmd}"
    elif has_cmd "${cmd}"; then
        resolved="$(command -v "${cmd}")"
    else
        return 1
    fi
    case "${resolved}" in
        *intel*oneapi*mpi*|*oneapi*/mpi/*|*impi*)
            return 0
            ;;
    esac
    "${resolved}" -V 2>&1 | head -n5 | grep -qi 'Intel.*MPI\|Intel(R).*MPI' && return 0
    "${resolved}" --version 2>&1 | head -n5 | grep -qi 'Intel.*MPI\|Intel(R).*MPI' && return 0
    return 1
}

resolve_launcher_candidate() {
    local candidate="$1"
    if [ -x "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if has_cmd "${candidate}"; then
        command -v "${candidate}"
        return 0
    fi
    return 1
}

MPI_COMPILE_ERR=""
probe_mpi_compile() {
    MPI_COMPILE_ERR=""
    local mpicc_bin
    mpicc_bin="$(resolve_mpi_tool mpicc mpiicc mpiicx || true)"
    [ -n "${mpicc_bin}" ] || {
        MPI_COMPILE_ERR="mpicc not found"
        return 1
    }

    local tmpdir src bin err rc
    tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/linstabpy-mpi-compile.XXXXXX" 2>/dev/null)" || {
        MPI_COMPILE_ERR="could not create temporary compile directory"
        return 1
    }
    src="${tmpdir}/mpi_probe.c"
    bin="${tmpdir}/mpi_probe"
    err="${tmpdir}/mpicc.err"
    cat > "${src}" <<'EOF'
#include <mpi.h>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
    return 0;
}
EOF
    if "${mpicc_bin}" "${src}" -o "${bin}" >"${err}" 2>&1; then
        rc=0
    else
        MPI_COMPILE_ERR="$(sed -n '1,3p' "${err}" | tr '\n' ' ' | sed -e 's/[[:space:]][[:space:]]*/ /g' -e 's/^ //g' -e 's/ $//g')"
        [ -n "${MPI_COMPILE_ERR}" ] || MPI_COMPILE_ERR="${mpicc_bin} failed to compile a program including mpi.h"
        rc=1
    fi
    rm -rf "${tmpdir}"
    return "${rc}"
}

# ── Header ──────────────────────────────────────────────────────────────────
echo ""
printf "=== ${COL_INFO}linstabpy install diagnostic${COL_RST} ===\n"
echo ""

# ── OS info ─────────────────────────────────────────────────────────────────
OS_NAME=$(uname -s)
KERNEL=$(uname -r 2>/dev/null || echo "unknown")
ARCH=$(uname -m 2>/dev/null || echo "unknown")
DISTRO="unknown"
if has_cmd lsb_release; then
    DISTRO="$(lsb_release -ds 2>/dev/null || echo "unknown")"
elif [ -f /etc/os-release ]; then
    DISTRO="$(awk -F= '/^PRETTY_NAME=/{gsub(/"/, "", $2); print $2; exit}' /etc/os-release 2>/dev/null || echo "unknown")"
fi
printf "  OS:        %s\n" "${DISTRO}"
printf "  Kernel:    %s\n" "${KERNEL}"
printf "  Arch:      %s\n" "${ARCH}"

# ── Package manager ────────────────────────────────────────────────────────
PM=""
if   has_cmd apt-get; then PM="apt"
elif has_cmd dnf;     then PM="dnf"
elif has_cmd yum;     then PM="yum"
elif has_cmd pacman;  then PM="pacman"
elif has_cmd zypper;  then PM="zypper"
fi
if [ -n "${PM}" ]; then
    printf "  Pkg mgr:   %s\n" "${PM}"
else
    printf "  Pkg mgr:   ${COL_MISS}not detected${COL_RST}\n"
fi
echo ""

# ── Required CLI tools ──────────────────────────────────────────────────────
printf "${COL_INFO}[tools]${COL_RST}\n"
for t in curl make cmake gcc g++ gfortran pkg-config; do
    check_tool "${t}"
done
check_any_mpi_tool mpicc mpicc mpiicc mpiicx
check_any_mpi_tool mpicxx mpicxx mpiicpc mpiicpx
check_any_mpi_tool mpifort mpifort mpif90 mpiifort mpiifx
check_tool mpiexec
check_tool h5cc optional
check_tool uv optional
echo ""

# ── Python ──────────────────────────────────────────────────────────────────
printf "${COL_INFO}[python]${COL_RST}\n"
if has_cmd python3; then
    PY3_PATH="$(command -v python3)"
    PY3_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "?")"
    printf "  %-14s ${COL_OK}OK${COL_RST}   %s  (%s)\n" "python3" "${PY3_PATH}" "${PY3_VER}"
else
    printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "python3"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi
if has_cmd uv; then
    UV_PATH="$(command -v uv)"
    UV_VER="$(uv --version 2>/dev/null | head -n1 || echo "?")"
    printf "  %-14s ${COL_OK}OK${COL_RST}   %s  (%s)\n" "uv" "${UV_PATH}" "${UV_VER}"
else
    printf "  %-14s ${COL_WARN}MISS${COL_RST} (will be auto-installed by install.sh)\n" "uv"
fi
echo ""

# ── MPI situation ───────────────────────────────────────────────────────────
printf "${COL_INFO}[mpi]${COL_RST}\n"

OPENMPI_FOUND=0
MPICH_FOUND=0
INTELMPI_FOUND=0
PREFER_INTELMPI=0
MPI_EXECUTORS=()

if [ -n "${LINSTAB2D_MPI_STACK:-}" ]; then
    printf "  Requested:   %s (LINSTAB2D_MPI_STACK)\n" "${LINSTAB2D_MPI_STACK}"
    case "${LINSTAB2D_MPI_STACK}" in
        intel|intelmpi|impi) INTELMPI_FOUND=1; PREFER_INTELMPI=1 ;;
        openmpi) OPENMPI_FOUND=1 ;;
        mpich) MPICH_FOUND=1 ;;
    esac
fi

if [ -n "${I_MPI_ROOT:-}" ]; then
    INTELMPI_FOUND=1
    printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  I_MPI_ROOT=%s\n" "${I_MPI_ROOT}"
elif has_cmd impi_info; then
    INTELMPI_FOUND=1
    printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  impi_info -> %s\n" "$(command -v impi_info)"
elif intel_mpi_available; then
    printf "  Intel MPI:   ${COL_INFO}AVAILABLE${COL_RST} oneAPI found, but not active in this shell\n"
fi

for candidate in mpiexec mpirun; do
    if mpi_cmd_looks_intel "${candidate}"; then
        INTELMPI_FOUND=1
        resolved="$(command -v "${candidate}")"
        MPI_EXECUTORS+=("${candidate}:${resolved}:intelmpi")
        printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
    fi
done

for candidate in mpiexec.openmpi mpirun.openmpi; do
    if has_cmd "${candidate}"; then
        OPENMPI_FOUND=1
        resolved="$(command -v "${candidate}")"
        MPI_EXECUTORS+=("${candidate}:${resolved}:openmpi")
        printf "  Open MPI:    ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
    fi
done

for candidate in mpiexec.mpich mpiexec.hydra mpirun.mpich; do
    if has_cmd "${candidate}"; then
        mpi_cmd_looks_intel "${candidate}" && continue
        MPICH_FOUND=1
        resolved="$(command -v "${candidate}")"
        MPI_EXECUTORS+=("${candidate}:${resolved}:mpich")
        printf "  MPICH:       ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
    fi
done

if has_cmd mpiexec; then
    generic_mpiexec="$(command -v mpiexec)"
    printf "  mpiexec:     ${COL_OK}FOUND${COL_RST}  %s" "${generic_mpiexec}"
    # Try to identify which stack the generic mpiexec belongs to
    mpiexec_id="$(mpiexec --version 2>&1 | head -n3 || true)"
    if echo "${mpiexec_id}" | grep -qi "Intel.*MPI\|Intel(R).*MPI"; then
        printf "  (Intel MPI)"
        INTELMPI_FOUND=1
    elif echo "${mpiexec_id}" | grep -qi "open.mpi\|OpenRTE\|ORTE"; then
        printf "  (Open MPI)"
        OPENMPI_FOUND=1
    elif echo "${mpiexec_id}" | grep -qi "mpich\|hydra\|HYDRA"; then
        printf "  (MPICH)"
        MPICH_FOUND=1
    fi
    printf '\n'
fi

if [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ] && [ "${INTELMPI_FOUND}" -eq 0 ]; then
    printf "  ${COL_MISS}No MPI installation detected.${COL_RST}\n"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

MPI_STACK_COUNT=$((OPENMPI_FOUND + MPICH_FOUND + INTELMPI_FOUND))
if [ "${MPI_STACK_COUNT}" -gt 1 ]; then
    if [ "${INTELMPI_FOUND}" -eq 1 ] && mpi_cmd_looks_intel mpiexec && mpi_cmd_looks_intel "$(resolve_mpi_tool mpicc mpiicc mpiicx 2>/dev/null || true)"; then
        printf "\n  ${COL_INFO}NOTE: Additional MPI launchers are visible, but the active mpicc/mpiexec are Intel MPI.${COL_RST}\n"
        printf "  Keep Intel MPI first in PATH while installing and running this environment.\n"
    else
        printf "\n  ${COL_WARN}WARNING: Multiple MPI stacks are visible in this environment.${COL_RST}\n"
        printf "  This causes ABI conflicts (e.g. mpi4py wheel links to one,\n"
        printf "  but mpiexec invokes the other). Remove one before installing.\n"
        WARNINGS+=("dual-mpi")
    fi
fi

# Probe which mpiexec can run -n 1 /bin/true
# Short timeout + forced kill to avoid hanging on broken MPI daemons.
echo ""
printf "  Probing mpiexec -n 1 /bin/true (5s timeout each):\n"
WORKING_MPIEXEC=""
MPI_PROBE_CANDIDATES=()
if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}/bin" ]; then
    MPI_PROBE_CANDIDATES+=("${I_MPI_ROOT}/bin/mpiexec" "${I_MPI_ROOT}/bin/mpirun")
fi
MPI_PROBE_CANDIDATES+=(mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpirun.openmpi mpiexec mpirun)
for candidate in "${MPI_PROBE_CANDIDATES[@]}"; do
    resolved="$(resolve_launcher_candidate "${candidate}" || true)"
    [ -n "${resolved}" ] || continue
    if has_cmd timeout; then
        if timeout -k 2 5 "${resolved}" -n 1 /bin/true >/dev/null 2>&1; then
            printf "    %-22s ${COL_OK}OK${COL_RST}\n" "${candidate}"
            [ -z "${WORKING_MPIEXEC}" ] && WORKING_MPIEXEC="${resolved}"
        else
            printf "    %-22s ${COL_MISS}FAIL/TIMEOUT${COL_RST}\n" "${candidate}"
        fi
    else
        printf "    %-22s ${COL_WARN}SKIP${COL_RST} (timeout command not available)\n" "${candidate}"
    fi
done
if [ -n "${WORKING_MPIEXEC}" ]; then
    printf "  Best candidate: %s\n" "${WORKING_MPIEXEC}"
else
    if [ "${OPENMPI_FOUND}" -eq 1 ] || [ "${MPICH_FOUND}" -eq 1 ] || [ "${INTELMPI_FOUND}" -eq 1 ]; then
        printf "  ${COL_MISS}No mpiexec passed the -n 1 /bin/true probe.${COL_RST}\n"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# libmpi.so check
echo ""
if has_cmd ldconfig; then
    libmpi_line="$(ldconfig -p 2>/dev/null | grep 'libmpi\.so\.' | head -n1 || true)"
    if [ -n "${libmpi_line}" ]; then
        printf "  libmpi.so:   %s\n" "${libmpi_line}"
    else
        printf "  libmpi.so:   ${COL_WARN}not found in ldconfig cache${COL_RST}\n"
    fi
fi
if resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null; then
    if probe_mpi_compile; then
        printf "  MPI compile: ${COL_OK}OK${COL_RST}   mpicc can compile and link mpi.h\n"
    else
        printf "  MPI compile: ${COL_MISS}FAIL${COL_RST} %s\n" "${MPI_COMPILE_ERR}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
else
    printf "  MPI compile: ${COL_WARN}SKIP${COL_RST} (mpicc not found)\n"
fi
echo ""

# ── HDF5 (optional) ────────────────────────────────────────────────────────
printf "${COL_INFO}[hdf5]${COL_RST} (optional, needed for h5py)\n"
if has_cmd h5cc; then
    h5ver="$(h5cc -showconfig 2>/dev/null | awk '/HDF5 Version:/{print $NF; exit}' || echo "?")"
    printf "  h5cc:        ${COL_OK}OK${COL_RST}   %s  (HDF5 %s)\n" "$(command -v h5cc)" "${h5ver}"
elif has_cmd pkg-config && pkg-config --exists hdf5 2>/dev/null; then
    h5ver="$(pkg-config --modversion hdf5 2>/dev/null || echo "?")"
    printf "  hdf5:        ${COL_OK}OK${COL_RST}   via pkg-config  (%s)\n" "${h5ver}"
elif has_cmd pkg-config && pkg-config --exists hdf5-serial 2>/dev/null; then
    h5ver="$(pkg-config --modversion hdf5-serial 2>/dev/null || echo "?")"
    printf "  hdf5-serial: ${COL_OK}OK${COL_RST}   via pkg-config  (%s)\n" "${h5ver}"
else
    printf "  hdf5:        ${COL_WARN}MISS${COL_RST} (not required unless LINSTAB2D_WITH_H5PY=1)\n"
fi
echo ""

# ── Remediation ─────────────────────────────────────────────────────────────
if [ "${FAIL_COUNT}" -gt 0 ] || [ "${#WARNINGS[@]}" -gt 0 ]; then
    printf "=== ${COL_WARN}ACTION REQUIRED${COL_RST} ===\n"
    echo ""

    # Dual-MPI warning
    for w in "${WARNINGS[@]}"; do
        if [ "${w}" = "dual-mpi" ]; then
            printf "  ${COL_WARN}Resolve MPI conflict (pick one):${COL_RST}\n"
            case "${PM}" in
                apt)
                    printf "    To use Intel MPI: unload/hide Open MPI and MPICH, then source oneAPI or load the Intel MPI module\n"
                    printf "    To keep Open MPI:  sudo apt remove mpich libmpich-dev && sudo apt autoremove\n"
                    printf "    To keep MPICH:     sudo apt remove libopenmpi-dev openmpi-bin && sudo apt autoremove\n"
                    ;;
                dnf|yum)
                    printf "    To use Intel MPI: unload/hide Open MPI and MPICH, then source oneAPI or load the Intel MPI module\n"
                    printf "    To keep Open MPI:  sudo %s remove mpich mpich-devel\n" "${PM}"
                    printf "    To keep MPICH:     sudo %s remove openmpi openmpi-devel\n" "${PM}"
                    ;;
                pacman)
                    printf "    To use Intel MPI: unload/hide Open MPI and MPICH, then source oneAPI or load the Intel MPI module\n"
                    printf "    To keep Open MPI:  sudo pacman -Rs mpich\n"
                    printf "    To keep MPICH:     sudo pacman -Rs openmpi\n"
                    ;;
                zypper)
                    printf "    To use Intel MPI: unload/hide Open MPI and MPICH, then source oneAPI or load the Intel MPI module\n"
                    printf "    To keep Open MPI:  sudo zypper remove mpich mpich-devel\n"
                    printf "    To keep MPICH:     sudo zypper remove openmpi openmpi-devel\n"
                    ;;
                *)
                    printf "    Remove one MPI stack using your package manager.\n"
                    ;;
            esac
            echo ""
        fi
    done

    # Missing tools
    if [ "${FAIL_COUNT}" -gt 0 ]; then
        printf "  ${COL_WARN}Install missing tools:${COL_RST}\n"
        case "${PM}" in
            apt)
                printf "    sudo apt update && sudo apt install -y build-essential cmake gfortran pkg-config curl ca-certificates libx11-dev\n"
                if [ "${INTELMPI_FOUND}" -eq 1 ] && { [ "${PREFER_INTELMPI}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; }; then
                    printf "    source /opt/intel/oneapi/setvars.sh --force   # or: module load <intel-mpi>\n"
                elif [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo apt install -y libopenmpi-dev openmpi-bin\n"
                else
                    printf "    sudo apt install -y mpich libmpich-dev\n"
                fi
                [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ] && printf "    sudo apt install -y libhdf5-dev\n"
                ;;
            dnf)
                printf "    sudo dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates libX11-devel\n"
                if [ "${INTELMPI_FOUND}" -eq 1 ] && { [ "${PREFER_INTELMPI}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; }; then
                    printf "    source /opt/intel/oneapi/setvars.sh --force   # or: module load <intel-mpi>\n"
                elif [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo dnf install -y openmpi openmpi-devel\n"
                else
                    printf "    sudo dnf install -y mpich mpich-devel\n"
                fi
                [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ] && printf "    sudo dnf install -y hdf5-devel\n"
                ;;
            yum)
                printf "    sudo yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates libX11-devel\n"
                if [ "${INTELMPI_FOUND}" -eq 1 ] && { [ "${PREFER_INTELMPI}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; }; then
                    printf "    source /opt/intel/oneapi/setvars.sh --force   # or: module load <intel-mpi>\n"
                elif [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo yum install -y openmpi openmpi-devel\n"
                else
                    printf "    sudo yum install -y mpich mpich-devel\n"
                fi
                [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ] && printf "    sudo yum install -y hdf5-devel\n"
                ;;
            pacman)
                printf "    sudo pacman -S --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates libx11\n"
                if [ "${INTELMPI_FOUND}" -eq 1 ] && { [ "${PREFER_INTELMPI}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; }; then
                    printf "    source /opt/intel/oneapi/setvars.sh --force   # or: module load <intel-mpi>\n"
                elif [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo pacman -S openmpi\n"
                else
                    printf "    sudo pacman -S mpich\n"
                fi
                [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ] && printf "    sudo pacman -S hdf5\n"
                ;;
            zypper)
                printf "    sudo zypper install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates libX11-devel\n"
                if [ "${INTELMPI_FOUND}" -eq 1 ] && { [ "${PREFER_INTELMPI}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; }; then
                    printf "    source /opt/intel/oneapi/setvars.sh --force   # or: module load <intel-mpi>\n"
                elif [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo zypper install openmpi openmpi-devel\n"
                else
                    printf "    sudo zypper install mpich mpich-devel\n"
                fi
                [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ] && printf "    sudo zypper install hdf5-devel\n"
                ;;
            *)
                printf "    Install manually: curl make cmake gcc g++ gfortran pkg-config, MPI dev tools, X11 headers\n"
                ;;
        esac
        if [ "${LINSTAB2D_WITH_H5PY:-0}" != "1" ]; then
            printf "  ${COL_INFO}Note:${COL_RST} HDF5/h5cc is optional; install HDF5 only when LINSTAB2D_WITH_H5PY=1.\n"
        fi
        if [ "${PREFER_INTELMPI}" -eq 0 ] && [ "${INTELMPI_FOUND}" -eq 0 ] && intel_mpi_available; then
            printf "  ${COL_INFO}Alternative:${COL_RST} Intel MPI is available but not active. To use it instead:\n"
            printf "    source /opt/intel/oneapi/setvars.sh --force\n"
            printf "    LINSTAB2D_MPI_STACK=intelmpi bash diagnose.sh\n"
            printf "    LINSTAB2D_MPI_STACK=intelmpi bash install.sh\n"
        fi
        echo ""
    fi
else
    printf "=== ${COL_OK}ALL CHECKS PASSED${COL_RST} ===\n"
    printf "  Ready to run: bash install.sh\n"
    if [ -n "${WORKING_MPIEXEC}" ]; then
        printf "  Working mpiexec: %s\n" "${WORKING_MPIEXEC}"
        printf "  Hint: LINSTAB2D_PETSC_MPIEXEC=%s bash install.sh\n" "${WORKING_MPIEXEC}"
    fi
    echo ""
fi

# Exit non-zero if there are critical failures or warnings so callers can gate on it.
if [ "${FAIL_COUNT}" -gt 0 ] || [ "${#WARNINGS[@]}" -gt 0 ]; then
    exit 1
fi

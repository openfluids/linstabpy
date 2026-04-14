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
check_tool mpicc
check_tool mpicxx
check_tool mpifort
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
MPI_EXECUTORS=()

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
    if echo "${mpiexec_id}" | grep -qi "open.mpi\|OpenRTE\|ORTE"; then
        printf "  (Open MPI)"
        OPENMPI_FOUND=1
    elif echo "${mpiexec_id}" | grep -qi "mpich\|hydra\|HYDRA"; then
        printf "  (MPICH)"
        MPICH_FOUND=1
    fi
    printf '\n'
fi

if [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; then
    printf "  ${COL_MISS}No MPI installation detected.${COL_RST}\n"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ "${OPENMPI_FOUND}" -eq 1 ] && [ "${MPICH_FOUND}" -eq 1 ]; then
    printf "\n  ${COL_WARN}WARNING: Both Open MPI and MPICH are installed.${COL_RST}\n"
    printf "  This causes ABI conflicts (e.g. mpi4py wheel links to one,\n"
    printf "  but mpiexec invokes the other). Remove one before installing.\n"
    WARNINGS+=("dual-mpi")
fi

# Probe which mpiexec can run -n 1 /bin/true
# Short timeout + forced kill to avoid hanging on broken MPI daemons.
echo ""
printf "  Probing mpiexec -n 1 /bin/true (5s timeout each):\n"
WORKING_MPIEXEC=""
for candidate in mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpirun.openmpi mpiexec; do
    has_cmd "${candidate}" || continue
    resolved="$(command -v "${candidate}")"
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
    if [ "${OPENMPI_FOUND}" -eq 1 ] || [ "${MPICH_FOUND}" -eq 1 ]; then
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
                    printf "    To keep Open MPI:  sudo apt remove mpich libmpich-dev && sudo apt autoremove\n"
                    printf "    To keep MPICH:     sudo apt remove libopenmpi-dev openmpi-bin && sudo apt autoremove\n"
                    ;;
                dnf|yum)
                    printf "    To keep Open MPI:  sudo %s remove mpich mpich-devel\n" "${PM}"
                    printf "    To keep MPICH:     sudo %s remove openmpi openmpi-devel\n" "${PM}"
                    ;;
                pacman)
                    printf "    To keep Open MPI:  sudo pacman -Rs mpich\n"
                    printf "    To keep MPICH:     sudo pacman -Rs openmpi\n"
                    ;;
                zypper)
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
                if [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo apt install -y libopenmpi-dev openmpi-bin\n"
                else
                    printf "    sudo apt install -y mpich libmpich-dev\n"
                fi
                printf "    sudo apt install -y libhdf5-dev   # optional, for h5py\n"
                ;;
            dnf)
                printf "    sudo dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates libX11-devel\n"
                if [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo dnf install -y openmpi openmpi-devel\n"
                else
                    printf "    sudo dnf install -y mpich mpich-devel\n"
                fi
                printf "    sudo dnf install -y hdf5-devel   # optional, for h5py\n"
                ;;
            yum)
                printf "    sudo yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates libX11-devel\n"
                if [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo yum install -y openmpi openmpi-devel\n"
                else
                    printf "    sudo yum install -y mpich mpich-devel\n"
                fi
                printf "    sudo yum install -y hdf5-devel   # optional, for h5py\n"
                ;;
            pacman)
                printf "    sudo pacman -S --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates libx11\n"
                if [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo pacman -S openmpi\n"
                else
                    printf "    sudo pacman -S mpich\n"
                fi
                printf "    sudo pacman -S hdf5   # optional, for h5py\n"
                ;;
            zypper)
                printf "    sudo zypper install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates libX11-devel\n"
                if [ "${OPENMPI_FOUND}" -eq 1 ] || { [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ]; }; then
                    printf "    sudo zypper install openmpi openmpi-devel\n"
                else
                    printf "    sudo zypper install mpich mpich-devel\n"
                fi
                printf "    sudo zypper install hdf5-devel   # optional, for h5py\n"
                ;;
            *)
                printf "    Install manually: curl make cmake gcc g++ gfortran pkg-config, MPI dev tools, X11 headers\n"
                ;;
        esac
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

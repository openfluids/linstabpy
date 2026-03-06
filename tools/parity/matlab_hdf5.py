"""Utilities for reading MATLAB v7.3 HDF5 exports with ``h5py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class ExtractedField:
    """Flattened leaf field extracted from a MATLAB HDF5 file."""

    path: str
    shape: tuple[int, ...]
    dtype: str


def sanitize_key(path: str) -> str:
    """Convert an HDF5 path into an NPZ-safe flat key."""
    return path.strip("/").replace("/", "__")


def unsanitize_key(key: str) -> str:
    """Recover the logical HDF5-style path from a flattened NPZ key."""
    return key.replace("__", "/")


def is_private_key(name: str) -> bool:
    return name.startswith("#")


def compound_to_complex(arr: np.ndarray) -> np.ndarray:
    """Convert MATLAB compound complex storage to a native complex array."""
    if arr.dtype.fields and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return np.asarray(arr["real"], dtype=float) + 1j * np.asarray(arr["imag"], dtype=float)
    return np.asarray(arr)


def decode_char_array(arr: np.ndarray) -> str:
    """Decode a MATLAB char array stored as uint8/uint16."""
    flat = np.asarray(arr).reshape(-1, order="F")
    chars = [chr(int(v)) for v in flat if int(v) != 0]
    return "".join(chars)


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, complex, np.integer, np.floating, np.complexfloating))


def _scalarize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return np.asarray(value).reshape(-1)[0].item()
    return value


def _decode_reference(ref: h5py.Reference, handle: h5py.File) -> Any:
    if not ref:
        return None
    node = handle[ref]
    if isinstance(node, h5py.Group):
        return {key: _decode_node(node[key], handle) for key in node.keys()}
    return _decode_node(node, handle)


def _decode_object_dataset(ds: h5py.Dataset, handle: h5py.File) -> Any:
    raw = np.asarray(ds[()], dtype=object)
    values = [_scalarize(_decode_reference(ref, handle)) for ref in raw.reshape(-1, order="F")]

    if all(v is None or isinstance(v, str) for v in values):
        return np.asarray(["" if v is None else v for v in values], dtype=str).reshape(raw.shape, order="F")

    if all(v is None or _is_scalar_like(v) for v in values):
        non_null = [v for v in values if v is not None]
        dtype: type[Any]
        if any(isinstance(v, (complex, np.complexfloating)) for v in non_null):
            dtype = complex
        elif any(isinstance(v, (float, np.floating)) for v in non_null):
            dtype = float
        elif all(isinstance(v, (int, np.integer)) for v in non_null):
            dtype = int
        else:
            dtype = object
        fill = "" if dtype is object else 0
        return np.asarray([fill if v is None else v for v in values], dtype=dtype).reshape(raw.shape, order="F")

    return np.asarray(values, dtype=object).reshape(raw.shape, order="F")


def _decode_node(node: h5py.Group | h5py.Dataset, handle: h5py.File) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _decode_node(node[key], handle) for key in node.keys()}

    ds = node
    if ds.dtype.kind == "O":
        return _decode_object_dataset(ds, handle)

    arr = compound_to_complex(np.asarray(ds))
    if arr.dtype.kind in {"S", "U"}:
        return arr
    if arr.dtype.kind in {"u", "i"} and arr.ndim >= 1 and arr.size > 0 and int(np.max(arr)) <= 65535:
        if ds.name.startswith("/#refs#/"):
            return decode_char_array(arr)
    return arr


def load_hdf5_tree(path: Path) -> dict[str, Any]:
    """Load public top-level MATLAB HDF5 nodes into a nested Python dict."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return {
            key: _decode_node(handle[key], handle)
            for key in handle.keys()
            if not is_private_key(key)
        }


def flatten_tree(tree: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[ExtractedField]]:
    """Flatten a nested tree into NPZ-safe arrays."""
    flat: dict[str, np.ndarray] = {}
    inventory: list[ExtractedField] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}/{key}" if prefix else key, child)
            return

        key = sanitize_key(prefix)
        if isinstance(value, str):
            arr = np.asarray([value], dtype=str)
        else:
            arr = np.asarray(value)
            if arr.dtype == object:
                raise TypeError(f"Object-valued field cannot be written to NPZ without pickle: {prefix}")
        flat[key] = arr
        inventory.append(ExtractedField(path=prefix, shape=tuple(int(s) for s in arr.shape), dtype=str(arr.dtype)))

    visit("", tree)
    return flat, inventory


def extract_file(path: Path) -> tuple[dict[str, np.ndarray], list[ExtractedField]]:
    """Load and flatten a MATLAB HDF5 export."""
    tree = load_hdf5_tree(path)
    return flatten_tree(tree)

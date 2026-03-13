# Plan 2026-03-13: Remaining Rename Cleanup for LinStabPy

Public-facing project naming already uses `LinStabPy`, while the current import path remains `linstab2d`. This file now tracks only the unfinished filesystem-level cleanup.

## Remaining work

1. Rename the local repository directory to `LinStabPy` if the user still wants the on-disk folder name changed.
2. After the directory rename, verify that path-sensitive scripts, paper builds, and local workflows still run from the new location.
3. Keep a full import-package rename out of scope unless explicitly requested later.

## Acceptance

1. The filesystem path is updated only if requested.
2. Local scripts and builds continue to work after the directory rename.
3. The implementation package remains `linstab2d` unless a separate package-rename task is opened.

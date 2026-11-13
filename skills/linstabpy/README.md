# linstabpy agent skill

This folder is an agent skill for [linstabpy](https://github.com/openfluids/linstabpy). An AI coding agent, such as Claude Code or Codex, loads it to learn how to work with LinStabPy: the install gate, the environment-variable CLI, the YAML case schema, the lazy Python API, and the NPZ output contract.

`SKILL.md` is the entry point the agent reads. `references/` holds the detail pages, and `references/generated/` is produced from the source by `scripts/generate_skill_refs.py` (CI runs it with `--check`).

## Install

Copy or symlink this folder into the skills folder of your agent. For Claude Code:

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD/skills/linstabpy" ~/.claude/skills/linstabpy
```

Run the command from the repository root. A worked session, with real output, is on the site: <https://linstabpy.vercel.app/claude-code/> (source: `docs/pages/claude-code.md`).

## Web index

An agent that reads web pages can start from <https://linstabpy.vercel.app/llms.txt>, an index of the documentation site. <https://linstabpy.vercel.app/llms-full.txt> holds the same pages in one file.

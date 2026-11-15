# RoboLab Skills

Agentic AI skills for automating common RoboLab workflows. Built on the [agentskill.io](https://agentskills.io) open standard -- works with Claude Code, Cursor, GitHub Copilot, and 30+ other AI-powered tools.

## Available Skills

| Skill | Description |
|-------|-------------|
| [robolab-taskgen](robolab-taskgen/) | Generate task files from natural language descriptions of robot manipulation goals |
| [robolab-init-pose-check](robolab-init-pose-check/) | Render randomized initial poses for new tasks and scenes, and review object groups |
| [robolab-scenegen](robolab-scenegen/) | Generate USD scene files from natural language descriptions |
| [robolab-github-pr](robolab-github-pr/) | Review, fix up, and merge external GitHub pull requests (DCO, SPDX, LFS, merge policy) |

## Skill Format

Each skill follows the [agentskill.io specification](https://agentskills.io/specification):

```
skills/
  <skill-name>/
    SKILL.md              # Required: YAML frontmatter + instructions
    references/           # Optional: detailed docs loaded on-demand
    scripts/              # Optional: executable helpers
    assets/               # Optional: templates, data files
```

The `SKILL.md` file contains YAML frontmatter (name, description, license, metadata) followed by markdown instructions that the AI agent follows.

## Using Skills

### Repository-local skill links

Skill sources live in `skills/`. To make a skill available to an agent that uses `.agents/skills/`, create a local discovery link:

```bash
# From the robolab repo root
mkdir -p .agents/skills
ln -s ../../skills/robolab-taskgen .agents/skills/robolab-taskgen
```

Configure your agent to load the public skill source or its repository-local link.

### Other AI Tools

Most tools that support the agentskill.io standard can load skills directly from the `skills/` directory. Refer to your tool's documentation for setup instructions.

## Adding a New Skill

1. Create a new directory under `skills/`:
   ```
   skills/my-skill/
     SKILL.md
     references/    # optional
   ```

2. Add YAML frontmatter to `SKILL.md`:
   ```yaml
   ---
   name: my-skill
   description: >
     What this skill does and when to use it.
   license: CC-BY-NC-4.0
   metadata:
     author: nvidia
     version: "1.0.0"
   ---
   ```

3. Write the skill instructions in markdown below the frontmatter.

4. Add the skill to the table in this README.

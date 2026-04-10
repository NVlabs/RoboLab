# RoboLab Skills

Agentic AI skills for automating common RoboLab workflows. Built on the [agentskill.io](https://agentskills.io) open standard -- works with Claude Code, Cursor, GitHub Copilot, and 30+ other AI-powered tools.

## Available Skills

| Skill | Description |
|-------|-------------|
| [robolab-taskgen](robolab-taskgen/) | Generate task files from natural language descriptions of robot manipulation goals |

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

### Claude Code

Symlink or copy skills into your project's `.claude/skills/` directory for auto-discovery:

```bash
# From the robolab repo root
ln -s ../../skills/robolab-taskgen .claude/skills/robolab-taskgen
```

Then invoke via slash command (`/robolab-taskgen`) or let Claude auto-trigger based on your request.

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

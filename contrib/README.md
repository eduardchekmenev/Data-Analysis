# contrib/

A space for user-contributed scripts, adaptations, and extensions that sit outside the core locked pipeline.

## Purpose

Use this folder for:
- Experiment-specific or project-specific variants of pipeline tools
- One-off analysis scripts that are useful to share but not intended for the main pipeline
- Adaptations for new instruments, biological systems, or experimental designs
- Utility scripts (data format converters, batch runners, plotting helpers, etc.)

## Guidelines

- Create a subfolder named after yourself or your project (e.g., `contrib/your-name/`, `contrib/project-name/`)
- Include a short README or header comment in each script describing its purpose, inputs, and outputs
- Keep contributions self-contained where possible — avoid breaking dependencies on the locked pipeline scripts
- Scripts here are community-maintained; the core maintainers do not guarantee support

## Relationship to the Main Pipeline

`Working Data Pipeline/` and `Raw Data Visualization/` are locked and maintained by the core team.
`contrib/` is the right place for anything that extends, wraps, or adapts those tools without modifying them directly.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Parallel Experimentation & Subagents
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- If there are 2+ theories supporting a project, branch them out and run in parallel
- Throw more compute at it to evaluate competing branches simultaneously
- Merge ONLY the best performing branch that supports the project to main, after completing all testings

### 3. Machine Learning Rigor
- After EACH update to a model, completely restart the training from scratch to prevent state leakage and overfitting
- Actively monitor and explicitly log checks for both underfitting and overfitting during the run
- Rigorously compare new model results against the base model baseline

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Run tests, check logs, verify CI/CD pipelines, and demonstrate correctness
- Ask yourself: "Would a staff engineer approve this?"

### 5. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

## Project Initialization

- **Auto-Setup**: For each new project, automatically create a GitHub repository and push the initial structure.
- **Pipeline First**: Set up CI/CD pipelines (e.g., GitHub Actions for websites, formatters, linters) at the very beginning of the project.
- **Sub-Task Rigor**: After every single sub-task, do all necessary things (lint, format, test) before pushing and committing to GitHub.

## Task Management

1. **Plan First**: Plan properly first, step-by-step. Write detailed specs to `tasks/todo.md` with checkable items.
2. **Implement Phase-Wise**: Verify the plan, create the to-do list, and start implementing phase-wise.
3. **Benchmark & Compare**: Once a phase is done, check its results and rigorously compare with the base model.
4. **Succeed & Push**: If it's an improvement, update `README.md` and `main.tex` accordingly, ensure CI/CD passes, then push and commit.
5. **Fail & Rollback**: If it is NOT an improvement, rollback the changes completely and go back to step 1.
6. **Capture Lessons**: Update `tasks/lessons.md` after any failures or corrections.

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
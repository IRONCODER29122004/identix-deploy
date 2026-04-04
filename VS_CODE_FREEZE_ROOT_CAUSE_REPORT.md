# VS Code Freeze Root Cause Report

Date: 2026-04-04
Workspace: Code_try_1

## Symptom
- VS Code UI freezes / "not responding" roughly every few minutes.
- Freezes persist after basic temp/cache cleanup.

## Deep Findings

### 1) Workspace scale is very high for real-time editor scanning
- Total files in workspace: 67,030
- Top-heavy folder:
  - Required: 66,882 files
- Root tracked files in Git index: 66,922

Impact:
- Explorer, search, file watchers, language servers, and SCM all pay high ongoing scan cost.

### 2) Git metadata and SCM pressure are significant
- Root Git metadata size (.git): ~4.043 GB
- Nested Git repository also exists:
  - Code_try_1/.git
  - Code_try_1/IDENTIX_V2_RELEASE/.git

Impact:
- VS Code Source Control and repository auto-detection process two repositories.
- Large tracked set causes repeated status refresh costs.

### 3) Nested repository detection increases extension-host work
- A nested repository under IDENTIX_V2_RELEASE was being detected in addition to root repo.

Impact:
- Extra SCM provider activity and duplicate refresh cycles.

### 4) Some large archive areas should not be indexed live
- Required and model/checkpoint trees are mostly archival/asset-heavy for editing purposes.
- Runtime-critical model paths still exist and were not removed.

## Actions Applied

### A) VS Code workspace performance profile updated
File updated: .vscode/settings.json

Added/updated:
- git.autofetch = false
- git.autorefresh = false
- git.autoRepositoryDetection = "openEditors"
- git.openRepositoryInParentFolders = "never"
- git.untrackedChanges = "hidden"
- git.ignoredRepositories = ["IDENTIX_V2_RELEASE"]
- files.exclude now hides Required/** and .git entry
- search.exclude now excludes Required/** and .git internals
- files.watcherExclude now excludes Required/** and .git internals
- python.analysis.diagnosticMode = "openFilesOnly"
- python.analysis.indexing = false
- python.analysis.exclude expanded to large/generated paths

Expected effect:
- Lower extension-host and file watcher churn.
- Less frequent full-graph updates in SCM and Python analysis.

### B) Local Git performance tuning applied (root repo)
Local git config set:
- feature.manyFiles = true
- core.untrackedcache = true
- core.fsmonitor = true
- gc.auto = 256

Measured git status latency:
- Before tuning: ~632.8 ms
- After tuning (repeat): ~428.7 ms (steady-state)

Expected effect:
- Faster status refresh and reduced blocking from Git operations.

## Safety / Integrity Notes
- No source code or model file deletions were made in this pass.
- No runtime-critical model checkpoints were removed.
- This pass focused on editor/SCM/indexing behavior only.

## Why freezing can still appear temporarily
- Existing VS Code process may still hold old watcher/index state until window reload.
- First reopen after config changes can still take initial settle time.
- Antivirus indexing on very large trees can still spike intermittently.

## Required next steps (safe)
1. Run: Developer: Reload Window.
2. If lag persists, fully close VS Code and reopen only this workspace root.
3. Keep only one repository active in Source Control (root).
4. Avoid opening Required as a separate root folder.

## Lean workspace profile added
- File: IDENTIX_LEAN_WORKSPACE.code-workspace
- Purpose: open only IDENTIX_V2_RELEASE and apply low-overhead editor settings.
- Benefit: avoids loading Required (66k+ files) in active VS Code window.

How to use:
1. File -> Open Workspace from File...
2. Select IDENTIX_LEAN_WORKSPACE.code-workspace
3. Restart VS Code once after opening.

## Optional advanced steps (if needed)
1. Move Required out of active workspace and re-open only app/deploy folders.
2. Run periodic maintenance:
   - git gc (when idle)
   - git maintenance run --auto
3. Add antivirus exclusions for workspace cache folders and .git objects (policy permitting).

## Verification checklist
- [x] .vscode performance settings applied
- [x] Nested repo ignored in VS Code SCM
- [x] Git many-files tuning enabled
- [x] Root-cause metrics recorded
- [x] Project files and models kept intact

# 📋 PROJECT RULES & GUIDELINES

**READ THIS BEFORE EVERY REQUEST**

This document contains the essential rules and guidelines for working with the IDENTIX project. Every response must follow these rules.

---

## 🎯 CORE RULES (MUST FOLLOW)

### Rule 1: Project Structure Respect
- ✅ **Required/** folder contains ALL active production code
- ✅ **Waste/** folder contains ONLY archives and historical references
- ✅ **Never move files between Required/ and Waste/ without explicit permission**
- ✅ **All new files created in Required/ folder**
- ✅ **All file names are DESCRIPTIVE and SELF-EXPLANATORY**

**File Naming Convention:**
```
Format: [Category]_[Purpose]_[Type].[extension]

Examples:
✅ main_app_flask.py (main Flask application)
✅ facial_landmark_detector_bisenet.py (facial landmark detection)
✅ deepfake_analyzer_multimethod.py (deepfake detection)
✅ video_processing_pipeline.py (video processing)
✅ database_connection_manager.py (MongoDB utilities)
✅ model_architectures_neural.py (model definitions)
✅ facial_recognition_landing.html (landing page template)
✅ landmark_detection_ui.html (landmark UI template)
```

### Rule 2: Code Path Consistency
- ✅ **All file paths must be CONSTANTS defined at the top of files**
- ✅ **Use relative paths starting from Required/ directory**
- ✅ **Example:**
  ```python
  # Good - at top of file
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  MODELS_DIR = os.path.join(BASE_DIR, 'models')
  DATA_DIR = os.path.join(BASE_DIR, 'data')
  UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
  ```

### Rule 3: File References in Code
- ✅ **When a file is renamed, UPDATE ALL IMPORTS immediately**
- ✅ **Use find-and-replace to catch all references**
- ✅ **Test after renaming to ensure no broken imports**
- ✅ **Document the rename in CHANGE_LOG.md**

**Import Update Examples:**
```python
# OLD - Don't use this
from deepfake_detector import DeepfakeDetector

# NEW - Use this when renamed
from deepfake_analyzer_multimethod import DeepfakeDetector
```

### Rule 4: Documentation Requirements
- ✅ **Every file must have a clear DOCSTRING at the top**
- ✅ **Docstring format:**
  ```python
  """
  Module: [DESCRIPTIVE NAME]
  Purpose: [What this file does in 1-2 sentences]
  Dependencies: [Key imports/requirements]
  
  Key Classes:
  - [ClassName]: [Brief description]
  
  Key Functions:
  - [function_name](): [Brief description]
  
  Usage:
  from module_name import ClassName
  obj = ClassName()
  """
  ```

### Rule 5: Code Organization Within Files
- ✅ **Imports at the top**
- ✅ **Constants below imports**
- ✅ **Functions/classes after constants**
- ✅ **Main execution at bottom (if applicable)**

### Rule 6: Template Files Naming
- ✅ **HTML files describe their PURPOSE not location**
- ✅ **Format: [Application]_[Feature]_[UI Type].html**

**Good Names:**
```
✅ facial_landmark_detection_interface.html
✅ deepfake_analyzer_dashboard.html
✅ video_segmentation_upload.html
✅ user_profile_settings.html
✅ legal_privacy_policy.html
```

**Bad Names:**
```
❌ landmark_index.html (too vague - which landmark?)
❌ results.html (results of what?)
❌ index.html (which index?)
❌ page.html (which page?)
```

### Rule 7: Static Files Organization
- ✅ **CSS in static/css/**
- ✅ **JavaScript in static/js/**
- ✅ **Images in static/images/**
- ✅ **Fonts in static/fonts/**
- ✅ **All with DESCRIPTIVE names**

### Rule 8: Scripts Organization
- ✅ **Utility scripts in scripts/ folder**
- ✅ **Naming: [Purpose]_[Action]_[Tool].py**

**Examples:**
```
✅ database_initialization_setup.py
✅ user_data_cleanup.py
✅ user_list_viewer.py
✅ prediction_debug_tool.py
```

### Rule 9: Tests Organization
- ✅ **All tests in tests/ folder**
- ✅ **Naming: test_[Feature]_[Scenario].py**

**Examples:**
```
✅ test_landmark_detection_simple.py
✅ test_mediapipe_accuracy_validation.py
✅ test_all_models_combinations.py
✅ test_video_sample_processing.py
```

### Rule 10: Documentation Files
- ✅ **All markdown files are DESCRIPTIVE**
- ✅ **Format: [Topic]_[Type]_[Scope].md**

**Examples:**
```
✅ RULES_PAGE.md (rules and guidelines)
✅ FILE_DESCRIPTIONS.md (file documentation)
✅ CONSOLIDATED_DOCUMENTATION.md (merged docs)
✅ facial_landmark_technical_guide.md
✅ deepfake_detection_explained.md
✅ deployment_production_guide.md
```

---

## 📊 RESPONSE RULES

### Rule 11: Response Format
- ✅ **Always update TODO list status**
- ✅ **Provide clear explanations of changes**
- ✅ **Show before/after for file renames**
- ✅ **List all files that were modified**
- ✅ **Provide verification/testing instructions**

### Rule 12: Change Documentation
- ✅ **Document every change in CHANGE_LOG.md**
- ✅ **Format: [Date] - [Change Type] - [Description]**
- ✅ **Include affected files**

### Rule 13: No Silent Changes
- ✅ **Never modify a file without showing the changes**
- ✅ **Always confirm changes with user before executing**
- ✅ **Show the old vs new side-by-side**

### Rule 14: Verification Steps
- ✅ **After renaming: Show what was renamed**
- ✅ **After code changes: Show the changes made**
- ✅ **After imports: List all imports updated**
- ✅ **Provide testing instructions**

---

## 🔄 WORKFLOW RULES

### Rule 15: Multi-Step Tasks
- ✅ **Break into logical steps**
- ✅ **Execute steps in correct order**
- ✅ **Verify after each step**
- ✅ **Update TODO list after each step**

### Rule 16: File Operations Order
1. **Read** (understand what needs changing)
2. **Plan** (list all changes needed)
3. **Confirm** (show user the plan)
4. **Execute** (make the changes)
5. **Verify** (test that changes work)
6. **Document** (update CHANGE_LOG.md)

### Rule 17: Dependency Handling
- ✅ **When renaming file A that B depends on: MUST update B**
- ✅ **Search for all imports of the old name**
- ✅ **Update ALL occurrences**
- ✅ **Test imports after changes**

### Rule 18: Error Prevention
- ✅ **Never leave broken imports**
- ✅ **Always verify paths after renaming**
- ✅ **Test code snippet after changes**
- ✅ **Use find-all-references to catch everything**

---

## 📝 DOCUMENTATION RULES

### Rule 19: File Description Format
Each file needs:
1. **File Name**: Current name
2. **New Name**: Proposed name (if renaming)
3. **Category**: Type of file
4. **Purpose**: What it does
5. **Dependencies**: What it imports/requires
6. **Key Components**: Classes, functions, etc.
7. **Status**: Active / Archive / Deprecated

### Rule 20: Change Log Format
```markdown
## [Date] - [Change Category]

**Changes Made:**
- [Old File Name] → [New File Name]
- Updated imports in: [File1], [File2]
- Updated paths in: [File3]

**Affected Files:**
- [File 1] (import updated)
- [File 2] (path updated)
- [File 3] (import updated)

**Testing:**
- [What to test]
```

### Rule 21: Track All Changes for Undo Capability ⭐ NEW
- ✅ **Note EVERYTHING you change**
- ✅ **Keep a detailed record of all modifications**
- ✅ **Purpose: Enable rollback if anything goes wrong**
- ✅ **What to track:**
  - Files created (with full path)
  - Files modified (with what changed)
  - Files deleted (with backup reference)
  - Code changes (before/after snippets)
  - Imports updated (old → new)
  - Paths changed (old → new)
  - Configuration changes

**Change Record Format:**
```markdown
## Change Record - [Date] [Time]

**Operation**: [Create/Modify/Delete/Rename]

**Files Affected:**
- [Filename]: [What changed]
- [Filename]: [What changed]

**Before:**
```code
[Original code/state]
```

**After:**
```code
[New code/state]
```

**To Undo:**
1. [Step-by-step undo instructions]
2. [Specific commands or reversions]

**Verification:**
- [How to verify change worked]
- [How to verify undo worked]
```

**Where to Track:**
- Create CHANGE_LOG.md in Required/ folder (if doesn't exist)
- Add entry for EVERY change session
- Include timestamp for chronological tracking
- Keep old entries (never delete history)

**Benefits:**
- ✅ Full audit trail of all work
- ✅ Easy to identify when issues started
- ✅ Clear path to undo problematic changes
- ✅ Knowledge transfer for future developers
- ✅ Debug aid when things break

### Rule 22: File Header Documentation ⭐ NEW
- ✅ **EVERY file must have a description at the top**
- ✅ **Update the description whenever you change the file**
- ✅ **Purpose: Understand file contents at a glance**

**Required Header Format:**

For **Python files** (.py):
```python
"""
═══════════════════════════════════════════════════════════════
FILE: [filename.py]
CREATED: [Date]
LAST MODIFIED: [Date] by [Who/What]
VERSION: [Version number]
═══════════════════════════════════════════════════════════════

PURPOSE:
[2-3 sentence description of what this file does]

KEY COMPONENTS:
- [Component 1]: [Description]
- [Component 2]: [Description]

DEPENDENCIES:
- [Package 1]: [Why needed]
- [Package 2]: [Why needed]

USAGE:
[How to use this file - import statements, run commands, etc.]

CHANGE HISTORY:
- [Date]: [Change description]
- [Date]: [Change description]

═══════════════════════════════════════════════════════════════
"""
```

For **HTML files** (.html):
```html
<!--
═══════════════════════════════════════════════════════════════
FILE: [filename.html]
CREATED: [Date]
LAST MODIFIED: [Date] by [Who/What]
VERSION: [Version number]
═══════════════════════════════════════════════════════════════

PURPOSE:
[2-3 sentence description of what this page does]

KEY FEATURES:
- [Feature 1]: [Description]
- [Feature 2]: [Description]

DEPENDENCIES:
- [CSS file]: [Purpose]
- [JS file]: [Purpose]
- [API endpoint]: [Purpose]

ROUTES:
- URL: [/route/path]
- Method: [GET/POST/etc.]

CHANGE HISTORY:
- [Date]: [Change description]
- [Date]: [Change description]

═══════════════════════════════════════════════════════════════
-->
```

For **JavaScript files** (.js):
```javascript
/**
 * ═══════════════════════════════════════════════════════════════
 * FILE: [filename.js]
 * CREATED: [Date]
 * LAST MODIFIED: [Date] by [Who/What]
 * VERSION: [Version number]
 * ═══════════════════════════════════════════════════════════════
 * 
 * PURPOSE:
 * [2-3 sentence description of what this file does]
 * 
 * KEY FUNCTIONS:
 * - [function1()]: [Description]
 * - [function2()]: [Description]
 * 
 * DEPENDENCIES:
 * - [Library 1]: [Why needed]
 * - [Library 2]: [Why needed]
 * 
 * USAGE:
 * [How to use - include in HTML, call functions, etc.]
 * 
 * CHANGE HISTORY:
 * - [Date]: [Change description]
 * - [Date]: [Change description]
 * 
 * ═══════════════════════════════════════════════════════════════
 */
```

For **Markdown files** (.md):
```markdown
<!--
═══════════════════════════════════════════════════════════════
FILE: [filename.md]
CREATED: [Date]
LAST MODIFIED: [Date] by [Who/What]
VERSION: [Version number]
═══════════════════════════════════════════════════════════════

PURPOSE:
[2-3 sentence description of this documentation]

SECTIONS:
- [Section 1]: [What it covers]
- [Section 2]: [What it covers]

AUDIENCE:
[Who should read this - developers, users, admins, etc.]

RELATED DOCS:
- [Doc 1]: [Relationship]
- [Doc 2]: [Relationship]

CHANGE HISTORY:
- [Date]: [Change description]
- [Date]: [Change description]

═══════════════════════════════════════════════════════════════
-->
```

**When to Update Header:**
- ✅ When creating a new file (add complete header)
- ✅ When modifying any code (update LAST MODIFIED)
- ✅ When adding features (update KEY COMPONENTS)
- ✅ When changing dependencies (update DEPENDENCIES)
- ✅ When fixing bugs (add to CHANGE HISTORY)
- ✅ When refactoring (update PURPOSE if scope changed)

**Benefits:**
- ✅ Instant understanding of file purpose
- ✅ Quick reference for dependencies
- ✅ Change history at a glance
- ✅ Version tracking
- ✅ Easier debugging
- ✅ Better collaboration

---

## 🚫 DO NOT DO

- ❌ **Don't rename files without updating imports**
- ❌ **Don't use vague file names (results.html, index.html)**
- ❌ **Don't move files between Required/ and Waste/ without permission**
- ❌ **Don't hardcode paths - use constants**
- ❌ **Don't leave documentation outdated**
- ❌ **Don't create files without description comments**
- ❌ **Don't skip testing after changes**
- ❌ **Don't modify files silently without showing changes**
- ❌ **Don't make changes without tracking them in CHANGE_LOG.md** ⭐ NEW
- ❌ **Don't create files without proper header documentation** ⭐ NEW
- ❌ **Don't update files without updating the header LAST MODIFIED date** ⭐ NEW

---

## ✅ DO DO

- ✅ **Always use descriptive file names**
- ✅ **Always update imports when renaming**
- ✅ **Always use path constants**
- ✅ **Always add module docstrings**
- ✅ **Always update CHANGE_LOG.md**
- ✅ **Always verify changes work**
- ✅ **Always show what changed**
- ✅ **Always get confirmation before major changes**
- ✅ **Always track changes for undo capability (Rule 21)** ⭐ NEW
- ✅ **Always add/update file headers when creating/modifying files (Rule 22)** ⭐ NEW
- ✅ **Always update LAST MODIFIED date in file headers** ⭐ NEW

---

## 📋 CHECKLIST FOR EVERY REQUEST

Before responding to any user request, check:

- [ ] **Have I read this RULES_PAGE.md?**
- [ ] **Do I understand the project structure?**
- [ ] **Have I identified all files that need updating?**
- [ ] **Do I have the correct file naming convention?**
- [ ] **Have I planned the changes in order?**
- [ ] **Will I update the TODO list?**
- [ ] **Will I show before/after changes?**
- [ ] **Will I test after making changes?**
- [ ] **Will I update CHANGE_LOG.md?**
- [ ] **Did I verify no broken imports?**
- [ ] **Will I track all changes with undo instructions (Rule 21)?** ⭐ NEW
- [ ] **Did I add/update file header documentation (Rule 22)?** ⭐ NEW

---

## 🎯 SUMMARY

**The Golden Rule**: Every file should be named so clearly that ANYONE reading the project can instantly understand what that file does, just by reading the name.

**Example of Good Naming:**
- `main_app_flask.py` → Obviously the main Flask application
- `facial_landmark_detector_bisenet.py` → Clearly detects facial landmarks using BiSeNet
- `deepfake_analyzer_multimethod.py` → Analyzes deepfakes using multiple methods
- `video_processing_pipeline.py` → Processes videos through a pipeline
- `database_connection_manager.py` → Manages database connections

**Example of Bad Naming:**
- `app.py` → Could be anything
- `deepfake_detector.py` → OK but not as clear
- `model.py` → What kind of model?
- `utils.py` → Too generic
- `main.py` → Could be anything

---

## 📌 LAST REMINDER

**BEFORE YOU GIVE ANY RESPONSE**: Read the relevant section of this RULES_PAGE.md that applies to the user's request, and make sure you follow EVERY RULE listed.

**If the user says "add a rule"**: Add that rule to this RULES_PAGE.md immediately.

**NEW RULES REMINDER**:
- ✅ **Rule 21**: Track all changes in CHANGE_LOG.md with undo instructions
- ✅ **Rule 22**: Add/update file headers with every file creation/modification

---

**Last Updated**: February 1, 2026
**Version**: 1.1.0 (Added Rules 21-22)
**Status**: Active - Read before every response


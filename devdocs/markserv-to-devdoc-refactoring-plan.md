# Project Rename: devdoc → devdoc

**Date:** 2025-10-16
**Context:** Complete project rename from "devdoc" to "devdoc"
**Status:** 📋 Planning Phase

## Executive Summary

This document outlines a comprehensive plan to rename the project from "devdoc" to "devdoc" across all files, directories, code references, documentation, and assets. The rename affects **221 occurrences** across **48 files** in the codebase.

## Rationale

**Why "devdoc"?**
- **Developer-focused**: Clear indication this is for developer documentation
- **Broader scope**: "dev" encompasses all developer workflows, not just markdown
- **Modern branding**: Shorter, cleaner name for CLI tool
- **Clear purpose**: "devdoc" immediately conveys documentation + development

**Migration Benefits**:
- Better SEO for developer documentation tools
- Clearer project identity
- Aligns with expanded feature set (MLIR, ONNX, dev commands)
- Modern branding for npm package

## Scope Analysis

### Impact Assessment

```
Total occurrences: 221
Files affected: 48
Categories:
  - Source code: 15 files
  - Documentation: 15 files
  - Test files: 10 files
  - Media assets: 27 files
  - Configuration: 5 files
  - Binary names: 2 files
```

### File Categories

#### 1. NPM Package & Configuration (Critical)
- `package.json` - Package name, bin commands, repository URLs
- `package-lock.json` - Dependency lock file (regenerated)
- `.beads/devdoc.db` - Project database

#### 2. Binary Executables (Critical)
- `bin/devdoc` - Main CLI entry point → `bin/devdoc`
- Binary references in package.json

#### 3. Source Code (Critical)
**TypeScript/JavaScript:**
- `src/cli/index.ts` - CLI bootstrap
- `src/cli/commands/serve.ts` - Serve command
- `src/cli/config.ts` - Configuration
- `src/cli/flags.ts` - Flag definitions
- `lib/server.ts` - Server implementation
- `lib/legacy/cli.ts` - Legacy CLI (archived)
- `lib/legacy/readme.ts` - Legacy readme script
- `scripts/build.js` - Build script

#### 4. Templates & Styles (High Priority)
- `lib/templates/directory.html` - Directory listing template
- `lib/templates/error.html` - Error page template
- `lib/templates/markdown.html` - Markdown rendering template
- `lib/templates/tracking.html` - Analytics template
- `lib/templates/devdoc.css` → `lib/templates/devdoc.css`

#### 5. Test Files (High Priority)
**Test Suites:**
- `tests/cli/config.test.ts`
- `tests/cli/serve.test.ts`
- `tests/unit/download.test.js`

**Test Fixtures:**
- `tests/fixtures/expected/devdoc-*.expected.html` (6 files)
- `tests/fixtures/directories/devdoc-cli-readme/`
- Test markdown files referencing devdoc

#### 6. Documentation (High Priority)
**Root Documentation:**
- `README.md` - Installation and usage guide
- `CHANGELOG.md` - Version history
- `CLAUDE.md` - Claude integration docs
- `AGENTS.md` - Agent documentation
- `GLOBAL_USAGE.md` - Usage instructions

**DevDocs Directory:**
- `devdocs/typescript-architecture.md`
- `devdocs/features.md`
- `devdocs/SIGNALS.md`
- `devdocs/claude-dev3000-integration.md`
- `devdocs/json-hero-integration-plan.md`
- `devdocs/mlir-location-info-support.md`
- `devdocs/logs/*.md` (5 files)

#### 7. Media & Assets (Medium Priority)
**Icons & Logos:**
- `lib/icons/devdoc.png` → `lib/icons/devdoc.png`
- `lib/icons/devdoc.svg` → `lib/icons/devdoc.svg`
- `lib/icons/devdoc-*.sketch` (3 files)
- `lib/icons/devdoc-*.svg` (2 files)

**Media Directory:**
- `media/devdoc-logo-*.{png,svg,sketch}` (12 files)
- `media/devdoc-*.{gif,png,svg}` (12 files)
- Demo images and screenshots

#### 8. CSS & Styling (Medium Priority)
- `lib/icons/icons.css` - Icon stylesheet references
- `lib/templates/devdoc.css` - Main stylesheet

#### 9. Third-Party Integrations (Low Priority)
- `third_party/jsonhero-web/` - JSON Hero build artifacts (regenerate)

#### 10. Configuration Files (Low Priority)
- `.claude/settings.local.json` - Claude Code settings

## Detailed Refactoring Plan

### Phase 1: Pre-Refactoring Preparation (30 minutes)

#### 1.1 Create Backup
```bash
# Create git branch for rename
git checkout -b refactor/rename-to-devdoc

# Tag current state
git tag pre-devdoc-rename
```

#### 1.2 Verify Clean State
```bash
# Ensure no uncommitted changes
git status

# Run tests to establish baseline
npm test

# Build to ensure everything works
npm run build
```

#### 1.3 Document Current State
- Capture current CLI output: `./bin/devdoc --help > pre-rename-cli-output.txt`
- List all test results: `npm test > pre-rename-test-results.txt`
- Screenshot package.json metadata

### Phase 2: Critical Infrastructure (1 hour)

#### 2.1 Rename Binary Executable
```bash
# Rename binary file
mv bin/devdoc bin/devdoc
```

**Files to update:**
- `package.json` - "bin" section
- `bin/devdoc` - Update internal references

#### 2.2 Update package.json
**Changes required:**
```json
{
  "name": "devdoc",
  "description": "🚀 Developer documentation server with AI model visualization",
  "bin": {
    "devdoc": "bin/devdoc",
    "readme": "bin/readme"
  },
  "homepage": "https://github.com/[username]/devdoc",
  "repository": {
    "type": "git",
    "url": "https://github.com/[username]/devdoc.git"
  },
  "bugs": {
    "url": "https://github.com/[username]/devdoc/issues"
  }
}
```

#### 2.3 Update CLI Bootstrap
**Files:**
- `src/cli/index.ts` - Update CLI name in comments and metadata
- `bin/devdoc` - Update CLI resolution paths and references

#### 2.4 Update Configuration
**Files:**
- `src/cli/config.ts` - Configuration loader references
- `src/cli/flags.ts` - Flag definition comments
- `.claude/settings.local.json` - Update project references

### Phase 3: Source Code Updates (2 hours)

#### 3.1 TypeScript/JavaScript Source Files

**Pattern-based replacements:**
```bash
# Command references in strings
"devdoc" → "devdoc"
"Devdoc" → "DevDoc"
"MARKSERV" → "DEVDOC"

# URL paths
"/devdoc" → "/devdoc"

# Package references
"@devdoc/" → "@devdoc/"
```

**Files requiring careful manual review:**
- `src/cli/commands/serve.ts` - Server command
- `lib/server.ts` - Server implementation (many references)
- `lib/splash.ts` - ASCII art (if contains "devdoc")
- `lib/legacy/cli.ts` - Legacy CLI (keep as-is for reference)
- `lib/legacy/readme.ts` - Legacy readme (keep as-is for reference)

#### 3.2 Template Files

**Files to update:**
- `lib/templates/directory.html` - "devdoc" references
- `lib/templates/error.html` - Error messages
- `lib/templates/markdown.html` - Page titles, meta tags
- `lib/templates/tracking.html` - Analytics references

**Specific updates:**
```html
<!-- Update page titles -->
<title>Devdoc</title> → <title>DevDoc</title>

<!-- Update meta tags -->
<meta name="generator" content="devdoc"> → <meta name="generator" content="devdoc">

<!-- Update class names (if needed) -->
class="devdoc-*" → class="devdoc-*"
```

#### 3.3 Stylesheet Renaming

```bash
# Rename main stylesheet
mv lib/templates/devdoc.css lib/templates/devdoc.css
```

**Update references in:**
- `lib/templates/*.html` - Update `<link>` tags
- `lib/server.ts` - Update file serving paths
- `lib/icons/icons.css` - Update any devdoc class references

### Phase 4: Test Suite Updates (1.5 hours)

#### 4.1 Rename Test Fixtures

```bash
# Rename expected output files
cd tests/fixtures/expected
for file in devdoc-*.expected.html; do
  mv "$file" "${file/devdoc/devdoc}"
done

# Rename test directories
mv tests/fixtures/directories/devdoc-cli-readme \
   tests/fixtures/directories/devdoc-cli-readme
```

#### 4.2 Update Test Files

**Files to update:**
- `tests/cli/config.test.ts` - Config loading tests
- `tests/cli/serve.test.ts` - Server command tests
- `tests/unit/download.test.js` - Download tests
- `tests/README.md` - Test documentation

**Update patterns:**
- Test descriptions: `"devdoc should..."` → `"devdoc should..."`
- CLI command references: `devdoc serve` → `devdoc serve`
- Expected output strings containing "devdoc"
- File path references to renamed fixtures

#### 4.3 Update Test Fixtures

**Markdown test files:**
- `tests/fixtures/markdown/*.md` - Update any devdoc references

**Expected HTML files:**
- Update all `devdoc-*.expected.html` files content
- Search/replace "devdoc" → "devdoc" in HTML content
- Update CSS class names if changed

### Phase 5: Documentation Updates (1 hour)

#### 5.1 README.md - Complete Rewrite

**Sections to update:**
- Title: `# Devdoc Installation Guide` → `# DevDoc Installation Guide`
- All command examples: `devdoc` → `devdoc`
- Installation instructions: `npm install -g devdoc` → `npm install -g devdoc`
- Package name references throughout

**Key changes:**
```markdown
# Before
$ npm install -g devdoc
$ devdoc serve ./docs

# After
$ npm install -g devdoc
$ devdoc serve ./docs
```

#### 5.2 CHANGELOG.md

Add new entry at top:
```markdown
## [3.0.0] - 2025-10-17

### BREAKING CHANGES
- **Project renamed from "devdoc" to "devdoc"**
  - CLI command changed: `devdoc` → `devdoc`
  - Package name changed: `devdoc` → `devdoc`
  - Binary executable renamed
  - All imports and references updated

### Migration Guide
Users should:
1. Uninstall old package: `npm uninstall -g devdoc`
2. Install new package: `npm install -g devdoc`
3. Update scripts: Replace `devdoc` with `devdoc`
4. Update documentation references
```

#### 5.3 Developer Documentation

**Files to update:**
- `CLAUDE.md` - Claude integration docs
- `AGENTS.md` - Agent documentation
- `GLOBAL_USAGE.md` - Usage instructions
- `devdocs/typescript-architecture.md` - Architecture docs
- `devdocs/features.md` - Feature documentation
- `devdocs/SIGNALS.md` - Signals list
- `devdocs/*.md` - All devdocs files

**Update patterns:**
- Code examples: `devdoc serve` → `devdoc serve`
- File paths: `/devdoc/` → `/devdoc/`
- Project references in prose

#### 5.4 Log Files

**Update recent logs:**
- `devdocs/logs/phase3-tooling-commands-implementation.md`
- `devdocs/logs/phase4-consolidation-completion.md`

**Add note to old logs:**
```markdown
> **Note**: This project has been renamed from "devdoc" to "devdoc".
> All references to "devdoc" in this historical log refer to the current "devdoc" project.
```

### Phase 6: Media & Assets (30 minutes)

#### 6.1 Icon Renaming

```bash
cd lib/icons
# Rename icon files
mv devdoc.png devdoc.png
mv devdoc.svg devdoc.svg
mv devdoc-color.svg devdoc-color.svg
mv devdoc-no-border.sketch devdoc-no-border.sketch
mv devdoc-color.sketch devdoc-color.sketch
```

#### 6.2 Media Directory Renaming

```bash
cd media
# Rename all devdoc-prefixed files
for file in devdoc-*; do
  mv "$file" "${file/devdoc/devdoc}"
done
```

**Files affected:**
- Logo variations (flat, term, wordmark, favicon)
- Demo GIFs and screenshots
- Banner images
- Sketch source files

#### 6.3 Update Icon CSS

**File:** `lib/icons/icons.css`
- Update background-image URLs: `devdoc.svg` → `devdoc.svg`
- Update class names if any reference devdoc

### Phase 7: Build & Integration (30 minutes)

#### 7.1 Update Build Scripts

**File:** `scripts/build.js`
- Update any devdoc references in build logic
- Update output paths if needed

#### 7.2 Third-Party Integration

**JSON Hero:**
- Regenerate build artifacts after rename
- Update references in `third_party/jsonhero-web/`

#### 7.3 Database Renaming

```bash
# Rename beads database
mv .beads/devdoc.db .beads/devdoc.db
```

### Phase 8: Verification & Testing (1 hour)

#### 8.1 Build Verification

```bash
# Clean build
npm run clean:artifacts

# Rebuild from scratch
npm run build

# Verify outputs
ls -la dist/src/cli/index.js
ls -la dist/lib/
```

#### 8.2 Test Suite

```bash
# Run full test suite
npm test

# Should show:
# - All tests passing
# - No references to "devdoc" in test output
# - Correct command names in examples
```

#### 8.3 CLI Verification

```bash
# Test new binary name
./bin/devdoc --help
# Should show: devdoc/3.0.0

# Test commands
./bin/devdoc serve --help
./bin/devdoc dev:build --help
./bin/devdoc graph:mlir --help

# Test legacy fallback
DEVDOC_USE_LEGACY_CLI=1 ./bin/devdoc serve --help
```

#### 8.4 Manual Testing Checklist

- [ ] CLI help displays correctly
- [ ] Serve command starts server
- [ ] Splash screen shows correct name
- [ ] HTML templates render with "DevDoc" branding
- [ ] CSS loads correctly with new filename
- [ ] Dev commands work (build, test, lint, clean)
- [ ] Graph commands work (mlir, onnx)
- [ ] Error pages show "DevDoc" branding
- [ ] Directory listings show correct branding
- [ ] Analytics tracking (if enabled) uses new name

### Phase 9: Package Lock Regeneration (15 minutes)

```bash
# Delete old package-lock
rm package-lock.json

# Regenerate with new package name
npm install

# Verify lock file is correct
grep "name" package-lock.json | head -1
# Should show: "name": "devdoc"
```

### Phase 10: Git Commit & Documentation (30 minutes)

#### 10.1 Stage All Changes

```bash
# Stage renamed files
git add -A

# Review changes
git status

# Check diff
git diff --cached --stat
```

#### 10.2 Commit with Detailed Message

```bash
git commit -m "refactor: rename project from devdoc to devdoc

BREAKING CHANGE: Project has been renamed from 'devdoc' to 'devdoc'

- Renamed npm package from 'devdoc' to 'devdoc'
- Renamed binary from 'bin/devdoc' to 'bin/devdoc'
- Updated all 221 code references across 48 files
- Renamed 27 media assets and icons
- Updated all documentation and examples
- Updated test fixtures and expectations
- Regenerated package-lock.json with new name

Migration:
- Users should uninstall 'devdoc' and install 'devdoc'
- Update scripts to use 'devdoc' command instead of 'devdoc'
- Update documentation references

Files changed:
- Source code: 15 files
- Documentation: 15 files
- Test files: 10 files
- Media assets: 27 files (renamed)
- Configuration: 5 files

See devdocs/devdoc-to-devdoc-refactoring-plan.md for complete details."
```

#### 10.3 Create Migration Tag

```bash
# Tag the rename commit
git tag v3.0.0-devdoc-rename

# Push branch and tag
git push origin refactor/rename-to-devdoc
git push origin v3.0.0-devdoc-rename
```

## Automation Scripts

### Search & Replace Script

```bash
#!/bin/bash
# File: scripts/rename-to-devdoc.sh

# Text replacements in code files
find . -type f \( -name "*.ts" -o -name "*.js" -o -name "*.json" -o -name "*.md" -o -name "*.html" \) \
  ! -path "*/node_modules/*" \
  ! -path "*/.git/*" \
  ! -path "*/dist/*" \
  ! -path "*/lib/legacy/*" \
  -exec sed -i '' 's/devdoc/devdoc/g' {} +

find . -type f \( -name "*.ts" -o -name "*.js" -o -name "*.json" -o -name "*.md" -o -name "*.html" \) \
  ! -path "*/node_modules/*" \
  ! -path "*/.git/*" \
  ! -path "*/dist/*" \
  ! -path "*/lib/legacy/*" \
  -exec sed -i '' 's/Devdoc/DevDoc/g' {} +

find . -type f \( -name "*.ts" -o -name "*.js" -o -name "*.json" -o -name "*.md" -o -name "*.html" \) \
  ! -path "*/node_modules/*" \
  ! -path "*/.git/*" \
  ! -path "*/dist/*" \
  ! -path "*/lib/legacy/*" \
  -exec sed -i '' 's/MARKSERV/DEVDOC/g' {} +

echo "✓ Text replacements complete"
```

### File Rename Script

```bash
#!/bin/bash
# File: scripts/rename-files.sh

# Rename binary
[ -f bin/devdoc ] && mv bin/devdoc bin/devdoc

# Rename CSS
[ -f lib/templates/devdoc.css ] && mv lib/templates/devdoc.css lib/templates/devdoc.css

# Rename test fixtures
cd tests/fixtures/expected
for file in devdoc-*.expected.html; do
  [ -f "$file" ] && mv "$file" "${file/devdoc/devdoc}"
done
cd ../../..

# Rename test directories
[ -d tests/fixtures/directories/devdoc-cli-readme ] && \
  mv tests/fixtures/directories/devdoc-cli-readme tests/fixtures/directories/devdoc-cli-readme

# Rename icons
cd lib/icons
for file in devdoc*; do
  [ -f "$file" ] && mv "$file" "${file/devdoc/devdoc}"
done
cd ../..

# Rename media
cd media
for file in devdoc-*; do
  [ -f "$file" ] && mv "$file" "${file/devdoc/devdoc}"
done
cd ..

# Rename database
[ -f .beads/devdoc.db ] && mv .beads/devdoc.db .beads/devdoc.db

echo "✓ File renames complete"
```

## Risk Assessment

### High Risk Areas

#### 1. Breaking Changes for Users
**Risk**: Users' scripts and workflows break
**Mitigation**:
- Clear migration guide in CHANGELOG
- Deprecation notice in old package (if possible)
- Comprehensive documentation
- Consider publishing both packages temporarily

#### 2. Test Suite Failures
**Risk**: Tests fail after rename
**Mitigation**:
- Run tests before and after
- Update test fixtures systematically
- Manual verification of CLI commands
- Regression testing

#### 3. Third-Party Integration
**Risk**: JSON Hero and other integrations break
**Mitigation**:
- Regenerate build artifacts
- Test integrations individually
- Update integration documentation

### Medium Risk Areas

#### 1. Asset References
**Risk**: Broken image/icon references
**Mitigation**:
- Comprehensive search for old paths
- Visual inspection of rendered pages
- Test directory listings

#### 2. CSS Class Names
**Risk**: Broken styling due to class name changes
**Mitigation**:
- Careful CSS review
- Visual testing of all page types
- Check responsive layouts

### Low Risk Areas

#### 1. Legacy CLI Files
**Risk**: Breaking legacy fallback
**Mitigation**:
- Keep lib/legacy/ unchanged
- Exclude from renaming scripts
- Test fallback mechanism

#### 2. Git History
**Risk**: Loss of searchability
**Mitigation**:
- Git tracks renames automatically
- Add migration notes to docs
- Tag before rename

## Rollback Plan

### If Issues Discovered

```bash
# Option 1: Revert to pre-rename tag
git reset --hard pre-devdoc-rename

# Option 2: Revert specific commit
git revert HEAD

# Option 3: Cherry-pick fixes
git checkout -b hotfix/devdoc-rename-fixes
# Apply fixes
git cherry-pick <commit-sha>
```

### Emergency Procedures

1. **Immediate rollback**: Reset to pre-rename tag
2. **Partial rollback**: Revert package.json only, keep code changes
3. **Forward fix**: Keep rename, fix individual issues
4. **Dual publication**: Publish both "devdoc" and "devdoc" packages

## Post-Refactoring Checklist

### Immediate Verification (Day 1)

- [ ] All tests passing (npm test)
- [ ] Build succeeds (npm run build)
- [ ] CLI help works (./bin/devdoc --help)
- [ ] All commands functional
- [ ] HTML renders correctly
- [ ] CSS loads properly
- [ ] Icons display correctly
- [ ] No console errors
- [ ] Git commit successful
- [ ] Documentation updated

### Extended Verification (Week 1)

- [ ] No broken links in documentation
- [ ] All examples tested manually
- [ ] Third-party integrations work
- [ ] GitHub repo renamed (if applicable)
- [ ] NPM package published (if applicable)
- [ ] Docker images updated (if applicable)
- [ ] CI/CD pipelines work
- [ ] Migration guide tested by user
- [ ] Old package deprecated
- [ ] Search engine results updated

### Long-term Monitoring (Month 1)

- [ ] User migration complete
- [ ] No reported issues
- [ ] Analytics showing new name
- [ ] Documentation indexed correctly
- [ ] Community awareness established
- [ ] Old references cleaned up

## Timeline Estimate

### Conservative Estimate (With Breaks)
- **Preparation**: 30 minutes
- **Critical Infrastructure**: 1 hour
- **Source Code**: 2 hours
- **Tests**: 1.5 hours
- **Documentation**: 1 hour
- **Media Assets**: 30 minutes
- **Build & Integration**: 30 minutes
- **Verification**: 1 hour
- **Package Lock**: 15 minutes
- **Git Commit**: 30 minutes

**Total**: ~8.5 hours (1 full work day)

### Aggressive Estimate (Using Scripts)
- Automated replacements: 1 hour
- Manual review: 2 hours
- Testing: 1.5 hours
- Documentation: 1 hour

**Total**: ~5.5 hours (half work day)

## Success Criteria

### Must Have
- ✅ All tests pass
- ✅ Build succeeds
- ✅ CLI works with new name
- ✅ No broken references
- ✅ Documentation updated

### Should Have
- ✅ Clean git history
- ✅ Migration guide complete
- ✅ All assets renamed
- ✅ Third-party integrations work

### Nice to Have
- ✅ Automated refactoring scripts
- ✅ Rollback procedures tested
- ✅ Community announcement prepared

## Appendix: Complete File List

### Files Requiring Updates (48 total)

#### Configuration (5)
1. `package.json`
2. `package-lock.json` (regenerate)
3. `.beads/devdoc.db`
4. `.claude/settings.local.json`
5. `tsconfig.json` (if contains references)

#### Source Code (15)
6. `src/cli/index.ts`
7. `src/cli/commands/serve.ts`
8. `src/cli/config.ts`
9. `src/cli/flags.ts`
10. `lib/server.ts`
11. `lib/splash.ts`
12. `lib/legacy/cli.ts` (skip - kept for reference)
13. `lib/legacy/readme.ts` (skip - kept for reference)
14. `lib/templates/directory.html`
15. `lib/templates/error.html`
16. `lib/templates/markdown.html`
17. `lib/templates/tracking.html`
18. `lib/templates/devdoc.css` (rename to devdoc.css)
19. `lib/icons/icons.css`
20. `scripts/build.js`

#### Documentation (15)
21. `README.md`
22. `CHANGELOG.md`
23. `CLAUDE.md`
24. `AGENTS.md`
25. `GLOBAL_USAGE.md`
26. `tests/README.md`
27. `devdocs/typescript-architecture.md`
28. `devdocs/features.md`
29. `devdocs/SIGNALS.md`
30. `devdocs/claude-dev3000-integration.md`
31. `devdocs/json-hero-integration-plan.md`
32. `devdocs/mlir-location-info-support.md`
33. `devdocs/logs/phase3-tooling-commands-implementation.md`
34. `devdocs/logs/phase4-consolidation-completion.md`
35. `devdocs/logs/sandbox-explanation.md`

#### Tests (10)
36. `tests/cli/config.test.ts`
37. `tests/cli/serve.test.ts`
38. `tests/unit/download.test.js`
39. `tests/fixtures/expected/*.expected.html` (6 files to rename)
40. `tests/fixtures/markdown/*.md` (4 files to update)
41. `tests/fixtures/directories/devdoc-cli-readme/` (rename)

#### Media & Assets (27)
42-48. `lib/icons/devdoc-*` (7 files)
49-68. `media/devdoc-*` (20 files)

## Conclusion

This refactoring plan provides a systematic approach to renaming the project from "devdoc" to "devdoc". By following the phases in order and using the provided automation scripts, the rename can be completed safely with minimal risk.

**Key Principles**:
1. **Systematic approach**: Follow phases in order
2. **Verification at each step**: Test after each major change
3. **Automation where possible**: Use scripts for repetitive tasks
4. **Manual review for critical files**: Don't blindly replace
5. **Rollback plan ready**: Tag before starting
6. **User migration support**: Clear documentation and guides

**Next Steps**:
1. Review and approve this plan
2. Create git branch for work
3. Execute Phase 1 (preparation)
4. Work through phases systematically
5. Verify at each stage
6. Complete with comprehensive testing

---

**Document Version**: 1.0
**Last Updated**: 2025-10-16
**Status**: Ready for execution

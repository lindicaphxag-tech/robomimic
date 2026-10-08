# PR submission packet — robomimic #270

**Target issue:** https://github.com/ARISE-Initiative/robomimic/issues/270

**One-click upstream compare / PR creation** (requires author's GitHub
browser login, not the restricted ChatGPT GitHub connector):
https://github.com/ARISE-Initiative/robomimic/compare/master...lindicaphxag-tech:robomimic:fix/add-delta-actions-converter-pr?expand=1

**Title:** `feat: convert robosuite absolute actions to delta actions`

## Body — paste into the upstream PR

Addresses #270, where maintainers invited an upstream contribution for
conversion between absolute and delta action conventions.

This patch adds an action-space translation utility for robosuite pose
trajectories, including explicit orientation and reference-frame
conventions instead of naïve subtraction of rotation coordinates.
The existing robosuite conversion/extraction entrypoints can opt into
the inverse action converter.

### Validation

- Exact *review branch* standalone public CI:
  https://github.com/lindicaphxag-tech/robomimic/actions/runs/37810928717
  — **5 passed** in 9.95s using Python 3.11, robosuite 1.5.2,
  MuJoCo 3.3.0, OSMesa CPU.
- It checked the two test files that exist in the proposed **single-commit,
  6-file** upstream candidate.
- All six candidate code/test Git blob SHAs are identical to the already
  validated research branch files. The CI run on the review branch
  checked out the exact candidate source plus only its standalone
  workflow.
- Broader real OSC physical consistency demonstrations are available
  separately at
  https://github.com/lindicaphxag-tech/robomimic/actions/runs/37713698317
  (15 tests) and
  https://github.com/lindicaphxag-tech/robomimic/actions/runs/37715504522
  (20 tests). These are not implied to have been run by the upstream CI.

### Scope

The conversion is controller/action-contract dependent. It does not
guarantee identical dynamic physical trajectories under unobserved
controller memory, joint-space nullspace settings, contact dynamics or
arbitrary impedance changes. No physical hardware validation is claimed.

I welcome maintainer feedback on the preferred integration point or
reducing the patch size. AI assistance was used in code analysis, test
preparation and PR writing, disclosed here.

**Repository to compare:** `lindicaphxag-tech/robomimic`
**Head branch:** `fix/add-delta-actions-converter-pr`
**Base:** `ARISE-Initiative/robomimic:master`

## Publication status

An attempt to create this pull request through the connected GitHub
integration returned HTTP **403 Resource not accessible by integration**.
**No upstream PR was created by that attempt.** This packet must be
published using the user's browser session.

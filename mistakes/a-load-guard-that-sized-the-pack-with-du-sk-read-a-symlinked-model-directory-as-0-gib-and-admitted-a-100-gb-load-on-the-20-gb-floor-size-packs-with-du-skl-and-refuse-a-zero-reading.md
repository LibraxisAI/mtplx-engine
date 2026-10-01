# A load guard that sized the pack with `du -sk` read a symlinked model directory as 0 GiB and admitted a 100 GB load on the 20 GB floor; size packs with `du -skL` and refuse a zero reading

**Symptom.** Every Flash-Next boot in the 2026-09-22 quiet window printed `GUARD: pack 0 GiB, free 82 GiB, need 20 GiB` and was admitted. The harness's model path is a symlink into `~/.mtplx/models`, and `du -sk <symlink>` reports the link, not the tree. The loads succeeded only because nothing else was resident; the guard exists because two overlapping Flash-Next loads panicked the Mac on 09-21.

**Cause.** `du -sk "$PACK"` without `-L`, and no check that the reading is plausible.

**Fix / rule.** Size a pack with `du -skL` (or `du -sk "$(readlink -f "$PACK")"`), and make the guard refuse when the reading is under 1 GiB: a zero-sized "pack" is a wrong path, never a safe one. Any guard that admits on a floor alone must say so in its output so the operator sees it.

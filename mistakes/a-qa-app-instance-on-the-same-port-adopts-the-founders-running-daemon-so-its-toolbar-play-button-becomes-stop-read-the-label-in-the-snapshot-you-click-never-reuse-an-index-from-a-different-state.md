# A QA app instance on the same port adopts the founder's running daemon, so its toolbar play button becomes Stop. Read the label in the snapshot you click, never reuse an index from a different state

**Symptom:** 2026-09-06 10:39, candidate app 2011032 (isolated bundle id,
its own settings file) was launched while the founder's app 2011020 had
its daemon on :8000. The candidate showed "Çalışıyor" (running) because
it probes the configured port and adopts whatever answers there. Its
toolbar button at accessibility index 4, which had been "Start MTPLX" in
every earlier snapshot, was now the Stop button. One AX press from a
snapshot filtered on the sidebar stopped the founder's daemon (health
closed, zero serve processes) and his chat lane went dark until it was
restarted from his app at 10:40.

**Cause:** An element index is a position in the tree, not an identity;
the play and stop buttons share the slot and swap on daemon state. The
click was issued from a snapshot whose query filter did not return the
button, so its label was never read. Two apps bound to the same port are
one daemon with two owners.

**Fix / rule:** Click a toolbar control only from a snapshot whose
returned rows include that control with the expected label (query the
label, for example "başlat"), and treat an index carried over from an
earlier snapshot as unknown. When a QA instance must coexist with the
founder's app, give it a different port in its settings file so it
cannot adopt or stop his daemon; otherwise never touch its start or stop
controls while his daemon is up.

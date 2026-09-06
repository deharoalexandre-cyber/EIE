# Frozen August EWS archive - scope notice

These files preserve the August 2026 experiment and its recorded failures.
They are historical artifacts, not the specification of the September EIE
consumed-weight runtime. Do not silently edit them: their hashes are part of
the recorded provenance.

The frozen timing engine sends real I/O into a side arena while FFNs still
consume resident weights. Timed "end-to-end" results cover decode with that
I/O, excluding loading and prefill. Logical per-layer SLRU allocation does not
establish physical consumed-cache VRAM savings.

Its SHA table is populated on first observation, not compared against a
pre-trusted signed index. File hashes can be rechecked today, but do not by
themselves prove when a protocol was registered relative to its experiment.

The seven checked protocol/engine/script Git blobs match their announced
digests. Ten prompt files named by the manifest, raw route/timing files and
parts of the build setup are not published. Some scripts retain local Windows
paths. This is a partial research archive, not a clone-and-run full replication
bundle. The verdict remains an author-reported historical result.

Run `python tests/claims/verify_evidence.py` from the repository root for the
offline inventory and arithmetic checks. CRLF worktree hashes may differ from
the frozen LF Git-blob hashes; the script checks the latter explicitly.

For actual consumed weights, use the [September report](../../ews-consumed-20260905.md)
and [current runtime guide](../../../ews/runtime-port.md).

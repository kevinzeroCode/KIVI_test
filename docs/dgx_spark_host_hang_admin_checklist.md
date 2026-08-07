# DGX Spark host hang — admin diagnostic checklist

This host (`spark-1260`) has now interrupted two separate unattended
LongBench ablation runs with the same qualitative signature: the
application log, the companion 5-minute monitor loop, and `journald`
itself all stop writing within a short window, but the machine's actual
`reboot` only happens hours later. No Python traceback, CUDA error, or OOM
message ever appears in either application log, and neither incident could
be further diagnosed from an unprivileged account — `dmesg`, `journalctl
-k`, and `/sys/fs/pstore` all require elevated access this account does not
have.

**This document does not claim the root cause is proven.** It records what
was observed from an unprivileged account and lists the commands an admin
(or anyone in the `adm`/`systemd-journal` groups, or with `sudo`) can run
to fill the evidence gap — ideally before the next long GPU run.

## Incident #1 — K16/V2 (2026-08-05)

- Launch: `2026-08-04 18:16`
- Application log stopped: `~2026-08-05 06:50` (11/15 tasks done, 10/200
  rows into `triviaqa`)
- Actual reboot: `~2026-08-06 05:50` — **~23 hours** after the freeze
- `triviaqa.jsonl.partial` was fully valid (10/10 parseable rows, 0 errors)
- Resumed safely afterward using the existing skip/resume/append logic;
  completed cleanly with 0 further errors

## Incident #2 — K4/V16 (2026-08-07)

- Launch: `2026-08-07 09:58:16`
- Application log stopped: `2026-08-07 11:42:40` (3/15 tasks done, 43/200
  rows into `hotpotqa`)
- Monitor log stopped: `2026-08-07 11:38:39` (last snapshot fully normal —
  GPU 82°C / 87W / 96% util, 23GiB RAM used, no thermal/memory trend)
- `journald`'s last entry for that boot was `2026-08-07 09:00:29` — **before
  this run even launched**, i.e. journal logging itself had already gone
  silent roughly an hour before the job started, while the job still ran
  successfully for another ~1h44m
- Actual reboot: `2026-08-07 13:55:29` (`uptime -s`) — **~2h13m** after the
  freeze
- Unlike Incident #1, `hotpotqa.jsonl.partial` had a **corrupted tail**:
  line 44 was 76 bytes of NUL (`\x00`) with no trailing newline — a
  filesystem artifact typical of a block that was allocated but never
  flushed before an abrupt stop. This was repaired with
  `scripts/repair_partial_jsonl.py` (see `EXPERIMENT_STATUS.md` and the
  recovery manifest under `outputs/recovery_backups/` for details); the
  43 valid rows were kept, the corrupted line removed, with a full backup
  of the pre-repair file.

## Cross-incident read

Both incidents share the same shape (app + monitor + journald all fall
silent together, actual reboot follows much later, no application-level
error). The two runs used *opposite* quantization directions (K16/V2 =
Value-only quantized; K4/V16 = Key-only quantized), which weakens — but
does not rule out — a single mixed-K/V code path as the sole explanation,
and correspondingly raises suspicion of a host/driver-level issue shared
across both runs. Without kernel-level evidence this remains a working
hypothesis (`RECURRENT_HOST_LEVEL_HANG_SUSPECTED`), not a conclusion.

## Commands for an admin to run (require `sudo` or `adm`/`systemd-journal` group membership)

These were **not** run by this session — they require elevated permissions
this account doesn't have, and per this project's standing safety
constraints, `sudo` is never invoked automatically.

```bash
# Full kernel ring buffer for the boot that ended in the incident.
# For the K4/V16 incident, that's boot -1 relative to whichever boot is
# current at the time this is run (`journalctl --list-boots` to confirm).
sudo journalctl -k -b -1

# Narrow to the signatures most likely to explain a silent host freeze:
sudo journalctl -k -b -1 \
  | grep -Ei "NVRM|Xid|oom|killed process|watchdog|hung task|lockup|pcie|aer|thermal"

# Full kernel log with human timestamps, same keyword filter (covers
# anything journald itself may have missed logging, given journald's own
# silence was part of both incidents).
sudo dmesg -T \
  | grep -Ei "NVRM|Xid|oom|watchdog|hung|lockup|pcie|aer"

# Persistent crash storage -- may hold a panic/oops record that survived
# the reboot even if journald didn't log it.
sudo ls -la /sys/fs/pstore
sudo grep -RniE "panic|watchdog|lockup|NVRM|Xid|OOM" /sys/fs/pstore 2>/dev/null
```

Specifically look for:

- an **NVIDIA Xid error code** (GPU driver/hardware fault)
- **OOM killer** invocation (even though monitor snapshots showed normal
  memory headroom right up to the freeze)
- **soft lockup** / **hard lockup** / **hung task** kernel warnings
- a **kernel panic** record
- **PCIe AER** (Advanced Error Reporting) events
- **thermal shutdown** or **power** events
- anything logged in the gap between journald's last entry and the actual
  reboot for each incident (see timestamps above)

If any of the above is found, update this document and
`EXPERIMENT_STATUS.md` with the confirmed root cause. If nothing turns up
after a genuine attempt, that's also worth recording — it would point more
strongly toward a hardware-level event with no kernel-visible trace (e.g. a
BMC/firmware-level watchdog reset) rather than a Linux-kernel-detectable
one.

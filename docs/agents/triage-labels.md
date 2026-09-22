# Triage labels

The strings used for the five canonical triage roles. The `triage` skill reads this file to know which label to apply at each state.

| Role | Label | Meaning |
|---|---|---|
| Needs triage | `needs-triage` | Newly filed, not yet assessed. |
| Needs more info | `needs-info` | Waiting on the reporter for detail. |
| Ready for an agent | `ready-for-agent` | Specified well enough for an AFK agent to pick up. |
| Ready for a human | `ready-for-human` | Needs a human decision, review, or manual step. |
| Won't fix | `wontfix` | Ruled out; will not be actioned. |

All five use the default strings — no overrides.

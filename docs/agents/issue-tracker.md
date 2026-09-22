# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

**Repo is the personal fork `Xingkai98/AIInfraGuide`.** Development happens on branches pushed to the `fork` remote; the upstream `origin` (`caomaolufei/AIInfraGuide`) is read-only for us. Run `gh` with `--repo Xingkai98/AIInfraGuide` (or from a clone whose default remote is the fork) so issues land on the fork, not upstream.

## Conventions

- **Create an issue**: `gh issue create --repo Xingkai98/AIInfraGuide --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo Xingkai98/AIInfraGuide --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo Xingkai98/AIInfraGuide --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo Xingkai98/AIInfraGuide --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo Xingkai98/AIInfraGuide --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo Xingkai98/AIInfraGuide --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When a skill says "publish to the issue tracker"

Create a GitHub issue on `Xingkai98/AIInfraGuide`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo Xingkai98/AIInfraGuide --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Destination / Notes / Decisions-so-far / Fog body. `gh issue create --repo Xingkai98/AIInfraGuide --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/Xingkai98/AIInfraGuide/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/Xingkai98/AIInfraGuide/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --repo Xingkai98/AIInfraGuide --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.

> **Note (verify on first use):** this gh client is v2.45.0. GitHub's native **sub-issues** and **issue-dependencies** APIs are newer and may not be exposed for this repo. Before relying on them, probe once:
> `gh api repos/Xingkai98/AIInfraGuide/issues/<n>/sub_issues` and `gh api repos/Xingkai98/AIInfraGuide/issues/<n>/dependencies/blocked_by`.
> A `404` on a real issue means the feature is unavailable here — fall back to the task-list + `Part of #<map>` convention for parenting, and a `Blocked by: #<n>` line for blocking.

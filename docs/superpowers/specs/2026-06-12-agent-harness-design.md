# Forge — Agent Harness & Team Management Platform (Design)

**Date:** 2026-06-12
**Status:** Approved design, pending implementation plan
**Working name:** Forge (rename freely)

## 1. Problem & vision

Managing multiple software projects with AI agents today means ad-hoc terminal sessions, no shared memory, no cost visibility, and no controlled execution environment. Forge is a self-hosted platform that lets one user (multi-user-ready) run **virtual dev teams** — role-based agents (team lead, architect, backend/frontend engineers, QA, devops) — against real repositories, driven from a **visual task board** of projects → epics → features → tasks.

Agents work autonomously inside **sandboxed Docker containers** with centrally managed secrets, permissions, skills, tools, MCP servers, and RAG access. Teams produce reviewable PRs, update **persistent memory** as they work, and run on **user-configurable models** with the harness picking sensible defaults per role.

### v1 success criterion

> Create a project pointing at a git repo, chat with a team-lead agent to turn an idea into a ticket on the board, hit **Run**, and watch a sandboxed team (lead + engineer + QA) produce a reviewed PR — with the merge gated on the human.

## 2. Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Agent engine | **Engine-agnostic `AgentRuntime` port**; default = headless coding agent (Claude Code / Agent SDK); OpenHands & CrewAI-Flows as future adapters | Coding-native agents decisively outperform generic framework agents at repo work; port keeps us unlocked |
| Model access | **LiteLLM gateway**, logical aliases per role, Claude as default models | Model-agnostic per user requirement; budgets/keys/cost tracking built in |
| Orchestration spine | **Temporal** (one workflow per run) | Durable pause-for-days human gates, crash-resume mid-run, retries; team already operates Temporal in llm_api. CrewAI rejected as spine: hierarchical mode documented-broken, ~3x token overhead, no durability (research: MAST study — role-played orchestration is where multi-agent dev teams fail) |
| Repo | **New standalone repo** | Harness manages many projects; must stand apart from any one of them |
| Conventions | **llm_api conventions are the default**: hexagonal `domain/ports/adapters/interactors`, FastAPI + Temporal + React/Vite/Tailwind + Postgres, `uv`, pytest, 80% coverage | Proven, operator already knows them |
| Users/auth | **Single-user now, multi-user-shaped** (Auth0, user/org columns from day one; local dev-user bypass) | Future-proof without real extra cost |
| Autonomy | **Configurable dial** per project, per-ticket override: `gated_all` (default) / `gated_merge` / `full_auto` | Autonomous teams fail expensively; build trust gradually |
| Task board | **Native board** in harness Postgres; GitHub stays code source of truth | Board is the primary human-agent interface; agent metadata doesn't fit GitHub/Jira |
| Board layout | **Kanban-first with slide-over ticket panel + toggleable team-chat rail** | User selection |
| Build order | **Phase A (end-to-end spine) → Phase C (management plane) → Phase B (full team)** | Prove the loop first, make complexity visible/debuggable, then scale the team |
| Deployment | **docker-compose, two profiles: local (laptop) & remote (server)**; k8s later | Local dev + remote operation both first-class |

## 3. Architecture

Two planes:

### Control plane (always running)

- **React board UI** → **FastAPI** (REST + SSE for live run events) → **Postgres**.
- **Temporal server + worker**: one workflow per ticket run. Human approvals = signals (wait indefinitely); board status = queries; crash = resume, never restart.
- **LiteLLM gateway**: all LLM traffic. Model aliases, virtual keys with budgets, spend tracking tagged `run_id`/`agent_role`/`stage`.
- **Egress proxy / credential broker**: the only path from sandboxes to the internet.

### Execution plane (ephemeral, per run)

- One hardened sandbox container per run, destroyed after PR: non-root, `cap-drop ALL`, `no-new-privileges`, read-only root FS + workspace volume, memory/CPU/pids limits, no docker socket, **zero secrets**.
- Workspace via **`WorkspaceProvider` port**: `GitCloneWorkspace` (remote: fresh clone, GitHub App token) or `LocalWorktreeWorkspace` (local: worktree off the user's checkout — agents work on local branches, instant editor visibility).
- Coding agent via **`AgentRuntime` port**: spawn/resume/stream-events/cancel. Default adapter wraps headless Claude Code / Agent SDK (`--output-format stream-json`, `--max-turns`, explicit `--allowedTools`, PreToolUse permission hooks).

### Deployment profiles

| | Local | Remote |
|---|---|---|
| Auth | dev-user bypass | Auth0 enforced, TLS |
| Workspace | LocalWorktreeWorkspace or clone | GitCloneWorkspace |
| Egress proxy | present, can toggle permissive | strict allowlist |
| Review | local branches in your editor | GitHub PRs |

Same compose file; profile = env config. Later k8s mapping: sandbox = Job / agent-sandbox CRD (gVisor when multi-tenant).

## 4. Domain model

Hierarchy: **Project → Epic → Feature → Task** (task = executable unit).

- **Project**: repo (GitHub URL or local path), team assignment, autonomy level, secrets, capability grants, budgets.
- **Work items**: markdown body, **structured acceptance criteria** (vague specs are the #1 multi-agent failure source), status (`Draft → Refining → Ready → In Progress → In Review → Approved → Done`, plus `Blocked`/`Failed`), activity feed.
- **Team**: named, reusable group of **AgentDefinitions**. Each: role (lead/architect/backend/frontend/qa/devops/custom), persona prompt, **model alias**, runtime adapter, capability grants (deny-by-default), memory scope.
- **Run**: one execution of a task by a team = one Temporal workflow. Records stages + timestamps, per-agent transcripts, per-agent/stage token+dollar cost, artifacts (branch, PR, QA report), budget state. Tasks may have many runs.
- **Refinement chat**: session with the team-lead agent attached to a project/epic; lead drafts epics/features/tasks onto the board live; nothing becomes `Ready` without the user in gated modes.
- **Governance**: `Secret` (encrypted, write-only), `SkillRegistryEntry`, `McpServer`, `ModelAlias`, `AuditEvent` (append-only).

## 5. Board UI

Kanban-first: board is the project home screen. Card click → **slide-over panel**: description, acceptance criteria, run timeline (stage-by-stage), live agent logs (SSE), per-agent cost, ticket-scoped team chat, approve/reject gate buttons. **Toggleable right rail** hosts the persistent team-lead chat (refinement and status flow through conversation while the board updates live).

Other screens: Teams (agents, roles, models, capabilities), Capabilities (skills/MCP/RAG/model registries), Secrets, Runs (cross-project history + costs), Spend dashboard, and a global **attention inbox** for pending gates/escalations. Epics/features render as board filters + a roadmap view.

## 6. Execution pipeline (one run)

Stages as Temporal activities/child workflows; ✋ = gate per autonomy dial:

1. **PLAN** — lead agent reads ticket + project memory → implementation plan. Architect agent reviews with a **different model** (cross-model review); ≤2 revise loops. ✋ plan gate (`gated_all`).
2. **PROVISION** — sandbox + workspace; mint 1-hour git token; create LiteLLM budget key for the run; assemble capability manifest.
3. **IMPLEMENT** — engineer agent(s) via AgentRuntime in sandbox; progress heartbeats stream to the board; commits to `agent/<task>` branch. Frontend+backend tasks may run parallel agents on split worktrees, lead merges.
4. **VERIFY** — QA agent in **fresh context** (sees ticket + diff, never the engineer transcript) adversarially tries to falsify "done": runs tests/build/lint, checks acceptance criteria. Fail → back to 3 with QA report; max 3 loops (configurable per project) then ✋ escalate. (Hallucinated completion ≈ 24% of multi-agent failures; never self-certify.)
5. **PR** — push branch; open PR (remote) or finalize local branch (local). PR body: plan, changes, QA evidence, cost report. ✋ **merge gate — always human unless `full_auto`**; GitHub branch rulesets enforce it server-side regardless (agent App cannot merge to main).
6. **LEARN** — curator agent (cheap model) distills run into memory diffs; sandbox destroyed, tokens revoked, audit sealed.

Devops role in v1 is thin: deterministic CI does the work; the agent only triages CI failures.

### Guardrails (every run)

- Budgets: max dollars, max turns, max wall-clock per stage → breach pauses + asks the user.
- **No-progress detector** on structured signals (new commits, distinct test results, files touched) — same failing tests N times or zero commits in M active minutes → escalate; never grind.
- **Blocked/infeasible** is a first-class agent action surfacing on the board (the Devin lesson).
- Every stage leaves disk artifacts (`plan.md`, `progress.md`, QA report) so retries/resumes read state from the workspace, not a lossy transcript.

### Supervision & liveness

No agent polls another agent; supervision is structural:

- The **workflow is the supervisor** — the lead is invoked per-stage and exits; persistent state lives in Temporal, which survives any process crash.
- **Worker liveness = activity heartbeats** (runtime adapter emits on every agent event). Heartbeat timeout → Temporal kills/retries the activity, resuming from `progress.md` + git state. Hard per-stage timeouts cap wall-clock.
- **Semantic stalling** is caught by the no-progress detector (mechanical signals, not LLM self-reports).
- **Harness liveness**: Docker `restart: always` + healthchecks (API, Temporal worker, proxy). Worker death mid-run → restarted worker resumes the workflow. A periodic **janitor workflow** reaps orphaned sandboxes/tokens.

## 7. Sandbox, secrets & permissions

- **Sandbox hardening** as listed in §3; egress deny-all except the proxy; per-project domain allowlist (git host, package registries, LiteLLM). Blocks metadata IP + RFC1918.
- **Secrets**: encrypted in Postgres, managed in UI, write-only. Agents see placeholders (`__github_token__`); the **credential-injecting proxy** substitutes real values only toward approved hosts, and redacts secrets from response bodies/logs (Infisical Agent Vault pattern). Git auth via credential helper → broker; tokens never in URLs/env/transcripts.
- **GitHub App** identity: per-repo install, `contents` + `pull_requests` write only; 1-hour installation tokens minted per run, revoked after. Rulesets on `main`: require PR + human review; App not on bypass list; optional push ruleset restricting App to `agent/*`.
- **Permissions**: per-role tool allowlists in config, enforced outside the model (PreToolUse-style interceptor in the runtime adapter). Risk tiers: workspace edits/tests = auto; push/install = auto + audited; credentials/out-of-workspace/force-push = blocked or human-approved. All decisions → append-only audit log, viewable per run.

## 8. Memory

Markdown-in-git, three scopes, human-editable (research consensus: files-in-git won for coding agents; vector/graph memory skipped in v1):

| Scope | Location | Contents |
|---|---|---|
| Project | managed repo: `AGENTS.md`/`CLAUDE.md` (≤~120 lines) + `docs/adr/` | conventions, architecture decisions, gotchas — shared by all agents |
| Role | harness memory repo: `roles/<role>.md` | cross-project role heuristics |
| Episodic | per-run `progress.md` in workspace | handoff state for resume/retry |

The **curator agent (Learn stage) is the only writer** to project/role memory: proposes additions *and deletions* as git commits → UI shows reviewable **memory diffs** (auto-applied in `full_auto`, gated otherwise). Write-time curation prevents memory rot.

Code search = grep/AST/LSP tools (no code embeddings — agentic search beat embeddings decisively). Phase B adds optional pgvector RAG over docs/ADRs/run summaries as a grantable capability.

## 9. Capabilities & model management

Four UI registries, enforced at PROVISION via the capability manifest:

- **Skills**: git repo of `SKILL.md` folders (open standard, portable across runtimes); per-role/team grants; UI authoring (edits = commits).
- **MCP servers**: approved-server registry, per-server tool allowlists (`mcp__server__tool`), credentials held by proxy/broker; deny-by-default; read-only sets for QA/review roles.
- **RAG indexes** (phase B): named pgvector indexes, grantable per role as a query tool.
- **Models**: provider credentials, model aliases, **role→alias defaults** (frontier: lead/architect/cross-review; mid: engineers; cheap: triage/QA-checks/curator) with per-agent/team/project overrides. Budgets at run / project-month / global-month. Cost roll-ups on tickets, runs, spend dashboard. LiteLLM version-pinned (post supply-chain incident); single instance is fine for single-user, scale behind an LB sharing Postgres if needed.

## 10. Build phases

**Phase A — the spine** (v1 criterion above): work-item CRUD + board + slide-over; refinement chat; one default team (lead+engineer+QA); pipeline with plan/merge gates; sandbox + egress proxy + GitHub App; Claude Code runtime adapter; LiteLLM static role defaults; project memory + progress files; local & remote profiles.

**Phase C — management plane**: secrets UI; skills/MCP registries; model config + budgets UI; audit log viewer; run inspector (transcripts/costs); autonomy dial UI; memory diff review UI.

**Phase B — full team**: all roles incl. richer architect/devops; custom roles; parallel engineers; role memory curation; second runtime adapter (OpenHands or CrewAI Flows); RAG indexes; chat-rail enhancements; multi-user RBAC.

## 11. Error handling

- Every external system (GitHub, LLM, Docker, Temporal) behind a port with typed domain errors; no silent swallowing.
- Temporal retry policies with capped attempts per activity; failures surface on the board card with the actual error.
- Runs always reach a terminal state (`done`/`failed`/`blocked`); janitor workflow reaps ungraceful deaths.
- User-facing errors are friendly; full context goes to server logs + audit trail.

## 12. Testing

- **Unit**: domain logic — pipeline state machine, budget math, permission rules, capability manifest assembly.
- **Integration**: API endpoints; Temporal workflows via its test framework (time-skipping for signals/timeouts/heartbeats).
- **Adapter tests with fakes**: `FakeAgentRuntime` scripting agent behavior — full pipeline tests without LLM calls; fake workspace/git fixtures.
- **E2E**: one real ticket against a fixture repo with a stub model through board → run → PR.
- 80% coverage gate in CI (llm_api standard).

## 13. Key research inputs

- MAST failure taxonomy (arXiv:2503.13657): 44% spec/role failures, 32% inter-agent misalignment, 24% verification failures → roles as verified pipeline stages, structured acceptance criteria, fresh-context QA.
- Anthropic: building effective agents / multi-agent research system / long-running harnesses → orchestrator-worker, ~15x token cost of multi-agent, initializer/coder/verifier pattern, progress-file resume.
- CrewAI: hierarchical mode documented-broken; ~3x token overhead; Flows-only if ever used → relegated to optional adapter.
- Sandboxing: GitHub Agentic Workflows firewall, Infisical Agent Vault, Claude Code sandbox-runtime → zero-secret containers + injecting egress proxy.
- Memory: Anthropic + Letta converged on markdown-in-git; grep beat embeddings for code search.
- Models: LiteLLM aliases/virtual keys/budgets; static per-role routing + overrides is the proven pattern.

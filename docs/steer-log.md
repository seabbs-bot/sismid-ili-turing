# Steer log

A running record of Sam's guidance (cleaned up) and the action taken in
response. Newest entries appended at the bottom. This is a living document;
Sam may add guidance at any time.

| # | Steer | Action taken |
|---|---|---|
| 1 | Build the model in Julia with Turing.jl; partial pooling across all 11 locations; a simple joint backfill model like baselinenowcast; test candidate formulations then submit the best. | Set project direction; created the Julia/Turing repo and brief. |
| 2 | Seasonal model with AR noise; seasonality and noise both partially pooled. | Recorded as the base model design. |
| 3 | Let seasonality vary by season (random effect) and by location (multivariate normal?). | Added as candidate axes (season RE; MVN vs independent location effects). |
| 4 | Consider AR as a VAR across locations. | Added VAR vs independent AR as a candidate axis. |
| 5 | Score WIS with ScoringRules.jl from the EpiAware org. | Verified `interval_score`/`quantile_score`; adopted for the WIS helper. |
| 6 | Seasonality from a random-effect structure (not Fourier), varying by location and time; test MV vs independent; test AR vs differencing, time-varying AR, and AR order > 2; repo with a report per experiment. | Captured in the brief and plan as candidate axes. |
| 7 | Tree/beam search: iterate, select by WIS, refine the winner, repeat. | Recorded the tree-search strategy. |
| 8 | Guard overfitting using WIS and the standard deviation of WIS. | Made mean WIS + WIS SD the selection metric. |
| 9 | Repo needs a README stating the ask and the goal. | Wrote README. |
| 10 | Use parallel subagents and implement/review loops; merge to main via PR or directly. | Recorded the workflow; later launched the agent fleet. |
| 11 | Ignore the red shared-box load; push ahead. | Set the compute-guard override to green. |
| 12 | MCMC is slow — screen with Pathfinder first; keep a few core jobs running regardless of load. | Adopted Pathfinder-first screening; core jobs run regardless. |
| 13 | Use Turing callbacks to monitor fits live. | Specified in the inference wrapper. |
| 14 | Use Mooncake as the reverse-mode AD backend for all models. | Set Mooncake as the AD backend everywhere; smoke test confirms it. |
| 15 | Also report WIS on the log scale, but do not optimise on it; flag when model choice would differ. | Added log-scale WIS as a report-only metric. |
| 16 | Repo on seabbs-bot, not nfidd. | Remote confirmed `seabbs-bot/sismid-ili-turing`; moved the local clone to `~/code/seabbs/`. |
| 17 | Docs in markdown, clearly structured: guidelines, then plan, then infra as it lands, then per-loop reports. | Wrote brief, plan, infrastructure, contracts, report template. |
| 18 | Backfill like baselinenowcast but with a non-monotonic report profile. | Updated the backfill design (revisions move up or down; no monotonic CDF). |
| 19 | Backfill differs by location and over time. | Added location/time variation and a backfill candidate axis. |
| 20 | Run at least 10 implement-and-review rounds. | Recorded in the plan and infrastructure. |
| 21 | Multiple implementers per review (lower-power models fine); the reviewer picks the preferred one. | Recorded; the fleet uses Sonnet implementers with review on the main thread. |
| 22 | Log transform is favoured; one order of differencing is likely wanted. | Set log as the default transform; differencing (d=1) is a favoured candidate. |
| 23 | Use the full Bayesian workflow: prior predictive, posterior predictive, residual checks. | Added a Bayesian-workflow section and a diagnostics module. |
| 24 | Do not cheat versus the session rules. | Added an experimental-integrity section: validation seasons only, vintage data, oracle scoring. |
| 25 | Check the session intent: tune many models, submit the best few. | Confirmed the plan matches the two session `qmd`s. |
| 26 | Make sure docs match the instructions; use a subagent. | Launched the docs-reconciler agent. |
| 27 | Do a submission smoke-test PR to learn the mechanics, in parallel. | Launched the submission-smoke agent (fork, validate, draft smoke PR). |
| 28 | Use 5-10 parallel subagents per wave; go fast. | Launched 11 agents across the wave. |
| 29 | Have a bot do EDA and write sequential reports the other bots can read. | Launched the EDA agent writing `docs/eda/`. |
| 30 | EDA should look for peak timing across seasons as inspiration, and keep iterating across the work. | Messaged the EDA agent to emphasise peak timing and treat reports as living inputs. |
| 31 | Reports can be pushed to git as they are made. | Reports are committed and pushed as they land. |
| 32 | All work in Julia. | Messaged agents to drop R; validator and future data work move to Julia. |
| 33 | Per-folder Project.toml files are allowed. | Noted; used where a sub-project helps. |
| 34 | The current front-runner model lives on git as a src module with everything needed to run and submit; experiments iterate around it so we are always ready to submit. | Set the architecture: `src/` is the always-submittable front-runner package; `experiments/` iterate and promote winners. |
| 35 | Keep a cleaned-up log of Sam's steer and the actions taken. | Created this document. |
| 36 | FlexiChains is the wanted chains backend; Turing already returns FlexiChains `VNChain` objects by default here, so embrace that rather than forcing MCMCChains. | Set FlexiChains as the chains type through inference and downstream code; no conversion to MCMCChains. |
| 37 | All implementation work is done by subagents; the orchestrator coordinates and reviews rather than writing code directly. | Recorded as a hard workflow rule: the orchestrating thread assigns, reviews, and merges, and does not write implementation code itself. |
| 38 | EDA is provisional and iterative, a look-review-restart loop, not a one-shot; do not over-index on the current EDA numbers, treat them as inspiration revisited across rounds. | Noted in the plan that EDA findings are a living input, re-checked each search round rather than fixed once. |
| 39 | Use EpiAware/EpiAwarePackageTools.jl for modular testing. | Adopted `EpiAwarePackageTools.jl` for the test suite scaffolding. |
| 40 | A Documenter docs page should render the Julia API docs and link all the markdown reports (brief, plan, infrastructure, contracts, steer-log, `docs/eda/`, and the per-loop reports) in one tidy nav. | Added a Documenter site as a tracked infrastructure piece, linking the API docs and every markdown doc. |
| 41 | The current front-runner model lives as a loadable `SismidILITuring` src module with everything needed to run and submit; experiments iterate around it. | Reaffirms steer 34; recorded again in the plan and README as the architecture: `src/` is the always-submittable front-runner, `experiments/` iterate and promote winners. |
| 42 | Do not give up on a model for being complex or hard to fit; fix segfaults, slow sampling, or awkward geometry (rewrite AD-unfriendly code, reparameterise, reduce size) instead of dropping the candidate. | Recorded in the plan: complexity alone is not grounds to drop a candidate from the search. |
| 43 | Push search rounds forward without waiting on slow or complex models; score candidates as they become ready, keep slow ones in flight, and overlap rounds rather than serialising on the slowest fit. | Recorded in the plan: rounds overlap and score asynchronously rather than blocking on the slowest candidate. |
| 44 | EDA should push useful plots, not just prose, so other agents can see the same visual evidence. | EDA now commits plots under `docs/eda/figures/`. |
| 45 | Keep all the docs up to date with the current state as work lands, not only with the original requirements. | This refresh pass ticks off Phase 0/1 in the plan, records Phase 2 Round 1, and updates the transform, segfault, and submission-tooling status across README, brief, plan, and infrastructure. |
| 46 | Be sure of the GitHub submission steps, and have a baseline first-pass model ready to track in the hub. | Building reproducible submission tooling (`scripts/submit.jl`, `docs/submission.md`), dress-rehearsed locally with the real base model output; the front-runner base model is the baseline tracked model, ready to submit once the go-ahead is given. |
| 47 | EDA is a continuous iterate-and-review loop, not one-shot; it keeps deepening the analysis and updating `docs/eda/` across every round, folding in each round's results and failure modes and re-checking earlier estimates on the full training set. | Added a "Continuous iteration and periodic review" subsection to the plan; EDA re-checks and updates `docs/eda/` every round rather than once. |
| 48 | The implement-and-review loop likewise iterates continuously across rounds, and every agent must periodically review the current repo docs (`docs/eda/`, `plan.md`, `brief.md`, `steer-log.md`) and the latest round/experiment reports, at minimum at the start of each round or task. | Recorded in the same plan subsection as a standing rule, so agents stay aligned with the newest findings rather than working from stale assumptions. |
| 49 | After each search round, an EDA/review pass should update `docs/eda/` and the plan with what was learned (which regions or times models fail on, what to probe next) before the next round's candidates are designed. | Added to the plan subsection as the between-round handoff step. |
| 50 | Go much wider and be more reactive; there is free compute on the box, so exploit it rather than fitting narrowly and serially. | Fits switched to parallel across the free cores; rounds run many candidates at once. |
| 51 | The slowness was cold subprocess-per-fit re-paying Turing/Mooncake time-to-first-fit every split; compile once per process and fit in parallel instead. | Baseline and round fits switched to a single compiled process (compile once, then loop/thread the fits). |
| 52 | Each search round should be several candidate agents in PARALLEL plus a separate REVIEWER agent that judges their findings, not one agent doing everything. | Round 1 restructured into per-candidate agents plus a reviewer that ranks and picks the winner. |
| 53 | Round agents should EXPLORE and make their own modelling choices, not just fit a fixed candidate or repeat the same test in parallel. | Shifting the round design to autonomous explorer agents, each owning a direction and choosing its own best variant. |
| 54 | The baseline submission, a simple version of the full ask, is the priority and must actually land on the hub. | Baseline driven with a compile-once single process; orchestrator opens the PR directly rather than relying on an agent that keeps going idle. |
| 55 | Keep the steer log on GitHub current, and keep an EDA agent continuously running and committing plots. | This entry; a continuous EDA agent (`eda-loop`) keeps adding plots, committed as they land. |

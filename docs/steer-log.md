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

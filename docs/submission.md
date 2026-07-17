# Submission runbook

How to get a model's forecasts into the SISMID ILI Forecasting Sandbox hub (`reichlab/sismid-ili-forecasting-sandbox`), tested end to end on this project.
See [`contracts.md`](contracts.md) for the forecast table schema and [`brief.md`](brief.md) for the experimental-integrity rules this must not break.

## What the hub actually requires

Every submission is two files, both derived from `model_id = <team_abbr>-<model_abbr>` (e.g. `nfidd-turing`):

- `model-output/<model_id>/<origin_date>-<model_id>.csv` — one file per forecast origin date.
  Columns, in this exact order: `origin_date, location, target, horizon, target_end_date, output_type, output_type_id, value`.
  There is no `model_id` column; the model is implied by the file's path.
- `model-metadata/<model_id>.yml` — at minimum `team_abbr`, `model_abbr`, `designated_model` (bool).
  `team_abbr` and `model_abbr` must each match `^[a-zA-Z0-9_+]+$`, be at most 25 characters, and concatenate (with a `-`) back to `model_id`.
  Other teams' files (e.g. `model-metadata/hist-avg.yml`, `model-metadata/kot-kot.yml`) show the optional richer fields (`model_contributors`, `methods`, `citation`).

Valid values, read fresh from `hub-config/tasks.json` in the hub clone so this never drifts:

- `target`: `"ili perc"` (only value).
- `horizon`: `1, 2, 3, 4`.
- `location`: `"US National"` plus `"HHS Region 1"` through `"HHS Region 10"` (11 total).
- `output_type`: `"quantile"`, with 23 required levels: `0.01, 0.025, 0.05, 0.10, ..., 0.90, 0.95, 0.975, 0.99`.
- `origin_date`: 139 valid Saturdays from `2015-10-17` to `2020-02-29`, one per week within five flu seasons (with summer gaps).
  `target_end_date` must equal `origin_date + 7 x horizon` days.
- `value`: non-negative, non-decreasing across quantile levels within each `(location, horizon)` pair.

The experimental-integrity rule in [`contracts.md`](contracts.md) still applies here: only fit and validate on validation seasons 1 and 2 (2015/16, 2016/17) until finalists are locked; the three test seasons are held out.

## The real gate: hub CI

Opening a PR against `reichlab/sismid-ili-forecasting-sandbox` triggers `.github/workflows/validate-submission.yaml`, named **"Hub Submission Validation (R)"** in PR checks.
It runs `hubValidations::validate_pr(..., skip_submit_window_check = FALSE)` — note it also checks the submission-window rules, which our local pre-flight checker below does not.
This CI run is the authoritative check; everything else here is a pre-flight to catch problems before it runs.

We proved this gate works on a real submission: PR [#59](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/59) ("[smoke test] nfidd-smoketest submission mechanics check") added a minimal `nfidd-smoketest` model whose CSV content was copied from `hist-avg`'s known-good `2015-11-14` file, and the CI check `validate-submission` passed in 2m36s.

## Local pre-flight: `scripts/validate_submission.jl`

Before ever opening a PR, check a prepared submission locally against the same rules (mirrored from `hub-config/tasks.json` and `hub-config/model-metadata-schema.json`, read fresh from the hub clone so they can't drift):

```sh
julia --project=scripts/validate scripts/validate_submission.jl <model_id> [hub_path]
# or via Taskfile:
task validate MODEL_ID=<model_id> HUB_PATH=<hub_path>
```

It runs in its own small environment (`scripts/validate/Project.toml`: CSV, DataFrames, JSON, YAML) so it never needs Turing, Mooncake, or Pathfinder.
It checks column names/order, `target`/`location`/`horizon`/`output_type` values, `target_end_date` arithmetic, quantile-level completeness per `(location, horizon)`, non-negative and monotone `value`, and the metadata YAML's required fields — everything except the submission-window check, which only the real CI enforces.

## Route A: fork, branch, PR (git + gh) — the tested route

This is what both PR #59 (the smoke test) and the dress rehearsal below used.

1. Fork once, reuse forever: `gh repo fork reichlab/sismid-ili-forecasting-sandbox --clone`.
   We already did this — the fork is `seabbs-bot/sismid-ili-forecasting-sandbox`, cloned locally at `~/code/external/sismid-ili-sandbox-fork`.
   Set `git config user.name "seabbs-bot"` and `user.email "signin@samabbott.co.uk"` in that clone once.
2. Branch off an up-to-date `main`: `git checkout main && git pull && git checkout -b <branch>`.
3. Produce the submission (see `scripts/submit.jl` below), or write it by hand into `model-output/<model_id>/` and `model-metadata/<model_id>.yml`.
4. Validate locally (`scripts/validate_submission.jl`, above) before committing anything.
5. `git add`, `git commit`, `git push -u origin <branch>`.
6. `gh pr create --repo reichlab/sismid-ili-forecasting-sandbox --base main --head seabbs-bot:<branch> --draft --title ... --body ...`.
   Use `--draft` for anything experimental; watch `gh pr checks <PR#> --repo reichlab/sismid-ili-forecasting-sandbox` for the `validate-submission` result.

## Route B: GitHub web upload (no git needed)

For a one-off file or a quick fix, GitHub's web UI supports adding files directly to a new branch without cloning anything:

1. On the fork (`github.com/seabbs-bot/sismid-ili-forecasting-sandbox`), navigate into `model-output/<model_id>/` (create the folder via "Add file -> Create new file" if it doesn't exist yet, since GitHub only creates folders that contain a file).
2. Use "Add file -> Upload files" and drag the CSV(s) in, and separately add/edit `model-metadata/<model_id>.yml` as a new file under `model-metadata/`.
3. At the bottom of the upload/edit page, choose "Create a new branch for this commit and start a pull request", then open the PR against `reichlab/sismid-ili-forecasting-sandbox`'s `main` branch.
4. The same `validate-submission` CI check runs on this PR exactly as it would on one opened via `gh` — there is no difference in what gets checked.

This route trades the local pre-flight step for convenience: nothing checks the file before it's uploaded, so a mistake only surfaces once the CI runs on the PR.
Prefer Route A (with `scripts/validate_submission.jl` run first) whenever git/gh are available.

## `scripts/submit.jl`: the reproducible driver

```sh
# Fit + write + validate locally; PRINTS (does not run) the push/PR commands.
julia --project=. scripts/submit.jl --model-id nfidd-turing

# Same, but actually push the branch and open the PR once ready.
julia --project=. scripts/submit.jl --model-id nfidd-turing --pr
```

It calls `produce_submission(...; write=true)` to fit the base model on every cross-validation split of the given `--seasons` (default `1,2`) and write `model-output/<model_id>/*.csv` plus `model-metadata/<model_id>.yml` into `--hub-path` (default our fork clone), then runs `scripts/validate_submission.jl` against the result in its own environment.
By default (`--pr` unset) it stops there and prints the exact `git`/`gh` commands for the commit, push, and PR, so a plain run never touches the network.
`--pr` refuses to run if `--hub-path`'s `origin` remote looks like the upstream hub itself rather than a fork, so it can never push straight to `reichlab/sismid-ili-forecasting-sandbox`.

Flags: `--model-id`, `--seasons`, `--hub-path`, `--ndraws`, `--dmax`, `--window-weeks`, `--branch`, `--pr` — see the script header for defaults and detail.

## Dress rehearsal: does our real base model pass?

Before this runbook existed, the only proof of the submission mechanics was PR #59 — a copy of an existing `hist-avg` file, which never touched our own model.
As a follow-up, we fit the actual `base_model` (the same `base_model` -> `fit_pathfinder` -> `forecast_quantiles` steps `produce_submission` calls internally) on two real validation-season origin dates, wrote the result into the fork clone on a local-only branch (`dress-rehearse-base`, never pushed), and ran `scripts/validate_submission.jl` against it.
See the session notes for the pass/fail result — this is the assurance the smoke test could not give, because it never exercised our own model's output.

## What's paused, deliberately

Pushing `dress-rehearse-base` and opening any new PR against `reichlab/sismid-ili-forecasting-sandbox` for a real model stays paused for explicit sign-off.
`scripts/submit.jl --pr` and Route A's later steps are ready to run the moment that go-ahead is given; nothing in this repository runs them automatically.

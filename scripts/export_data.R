#!/usr/bin/env Rscript
# Export the course data objects to CSV for use in Julia.
# Reads the .rda files from the SISMID forecasting course package and writes
# flat CSVs into data/. Run from the repo root:
#   Rscript scripts/export_data.R

suppressMessages({
  library(dplyr)
})

course <- path.expand("~/code/nfidd/sismid-forecasting/data")
outdir <- path.expand("~/code/seabbs/sismid-ili-turing/data")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

objs <- c(
  "flu_data_hhs",
  "flu_data_hhs_versions",
  "flu_data_hhs_tscv_season1",
  "flu_data_hhs_tscv_season2",
  "flu_data_hhs_tscv_season3",
  "flu_data_hhs_tscv_season4",
  "flu_data_hhs_tscv_season5"
)

for (obj in objs) {
  rda <- file.path(course, paste0(obj, ".rda"))
  if (!file.exists(rda)) {
    message("skip (missing): ", rda)
    next
  }
  e <- new.env()
  load(rda, envir = e)
  df <- get(obj, envir = e) |>
    tibble::as_tibble() |>
    as.data.frame()
  out <- file.path(outdir, paste0(obj, ".csv"))
  write.csv(df, out, row.names = FALSE)
  message("wrote ", out, "  (", nrow(df), " rows, ",
          ncol(df), " cols)")
}

message("done")

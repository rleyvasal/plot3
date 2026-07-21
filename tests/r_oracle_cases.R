# ggplot2 / base-R oracle for plot3 semantic checks.
# Invoked as: Rscript r_oracle_cases.R <case_name>
# Prints one JSON document to stdout.

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required for plot3 R oracle")
  }
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r_oracle_cases.R <case>")
case_name <- args[[1]]

to_json <- function(x) {
  jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = 12)
}

# ── fixed datasets (match Python fixtures) ──────────────────────────────────
set.seed(1)
box_a <- rnorm(40, 0, 1)
box_b <- c(rnorm(39, 2, 1), 12.0)

cars <- data.frame(
  mpg = c(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4),
  cyl = c(6, 6, 4, 6, 8, 6, 8, 4),
  wt  = c(2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19)
)

# Tukey box stats via base boxplot.stats (same fences as ggplot2 default coef=1.5)
box_stats_one <- function(x, coef = 1.5) {
  s <- boxplot.stats(x, coef = coef)
  list(
    ymin = s$stats[[1]],
    lower = s$stats[[2]],
    middle = s$stats[[3]],
    upper = s$stats[[4]],
    ymax = s$stats[[5]],
    outliers = as.numeric(s$out)
  )
}

result <- switch(
  case_name,
  boxplot_tukey = list(
    a = box_stats_one(box_a),
    b = box_stats_one(box_b)
  ),
  hist_breaks = {
    h <- hist(cars$mpg, breaks = 5, plot = FALSE)
    list(
      breaks = as.numeric(h$breaks),
      counts = as.numeric(h$counts),
      mids = as.numeric(h$mids)
    )
  },
  density_eval = {
    d <- density(cars$mpg, n = 64, kernel = "gaussian")
    list(
      x = as.numeric(d$x),
      y = as.numeric(d$y),
      bw = d$bw,
      n = d$n
    )
  },
  quantiles = {
    list(
      mpg_q = as.numeric(quantile(cars$mpg, probs = c(0, 0.25, 0.5, 0.75, 1), type = 7)),
      wt_q = as.numeric(quantile(cars$wt, probs = c(0.25, 0.5, 0.75), type = 7))
    )
  },
  stop(sprintf("unknown oracle case: %s", case_name))
)

cat(to_json(result), "\n")

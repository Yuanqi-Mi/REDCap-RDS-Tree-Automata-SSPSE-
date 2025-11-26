#!/usr/bin/env Rscript
# ======================================================
# compute_sspse.R — Streamlit-Compatible SS-PSE Posterior + Visibility Plots
# Author: Yuanqi Mi  (fixed visibility = TRUE version)
# ======================================================

# ------------------------------
# 0. Load Required Packages
# ------------------------------
suppressPackageStartupMessages({
  if (!requireNamespace("sspse", quietly = TRUE)) {
    install.packages("sspse", repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace("RDS", quietly = TRUE)) {
    install.packages("RDS", repos = "https://cloud.r-project.org")
  }
  library(RDS)
  library(sspse)
})

# ------------------------------
# 1. Read Command-line Arguments
# ------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript compute_sspse.R <input.csv> <output.csv> <median_prior_N>")
}

input_csv  <- args[1]
output_csv <- args[2]
N_prior    <- as.numeric(args[3])

cat("===========================================\n")
cat(" SS-PSE Posterior Population Size Estimation\n")
cat("===========================================\n")
cat("Input file:  ", input_csv, "\n")
cat("Output file: ", output_csv, "\n")
cat("Median prior population size:", N_prior, "\n\n")

# ------------------------------
# 2. Load and Prepare Data
# ------------------------------
data <- read.csv(input_csv, stringsAsFactors = FALSE)

# 把 seed 的 recruiter 统一成 "0"
data$recruiter[data$recruiter == "seed"] <- "0"

rds <- as.rds.data.frame(
  data,
  id            = "id",
  recruiter.id  = "recruiter",
  network.size  = "network.size"
)
cat("✅ Loaded RDS dataset successfully (n =", nrow(rds), ")\n\n")

# ------------------------------
# 3. Run SS-PSE Model  (visibility = TRUE!!!)
# ------------------------------
set.seed(1234)
RNGkind("L'Ecuyer-CMRG")
options(mc.cores = 1)

cat("Running SS-PSE estimation with visibility model...\n")
fit <- posteriorsize(
  rds,
  median.prior.size = N_prior,
  visibility        = TRUE,   # ★ 关键：开启 visibility SS-PSE
  samples           = 1000,   # MCMC 后验样本数（可以按需要调大）
  interval          = 10      # 每 10 次保存一个样本（thin）
)
cat("✅ Model fitting complete.\n\n")

# ------------------------------
# 4. Summary Table (90% HPD)
# ------------------------------
summ <- summary(fit, HPD.level = 0.9)
summ_df <- as.data.frame(summ)
write.csv(summ_df, file = output_csv, row.names = TRUE)
cat("✅ Summary table saved to:", output_csv, "\n\n")
print(summ_df)

# ------------------------------
# 5. Posterior Plot (PDF + PNG)
# ------------------------------
plot_filename_base <- sub("\\.csv$", "", basename(input_csv))
pdf_filename <- paste0(plot_filename_base, "_sspse_posterior_N.pdf")
png_filename <- "sspse_plot.png"

cat("Generating posterior plot...\n")

# ---- PDF ----
pdf(pdf_filename, width = 10, height = 6)
par(mar = c(4, 4, 2, 8))  # extra margin for legend
plot(fit, type = "N")      # 内置 posterior 图
dev.off()

# ---- PNG ----
png(png_filename, width = 900, height = 500, res = 120)
par(mar = c(4, 4, 2, 8))
plot(fit, type = "N")
dev.off()

cat("✅ Posterior plot saved:", pdf_filename, "and", png_filename, "\n\n")



# ------------------------------
# 7. Done
# ------------------------------
cat("🎯 SS-PSE analysis completed successfully.\n")
cat("📊 Summary table:", output_csv, "\n")
cat("🖼️ Posterior figure:", pdf_filename, "and", png_filename, "\n")
cat("===========================================\n\n")









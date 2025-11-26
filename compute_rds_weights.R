#!/usr/bin/env Rscript
library(RDS)

args <- commandArgs(trailingOnly = TRUE)
input_csv  <- args[1]
output_csv <- args[2]
N          <- as.numeric(args[3])

data <- read.csv(input_csv, stringsAsFactors = FALSE)

# ---- FIX: All seeds should have recruiter.id = "0"
data$recruiter[data$recruiter == "seed"] <- "0"

# Create RDS object
rds <- as.rds.data.frame(
  data,
  id = "id",
  recruiter.id = "recruiter",
  network.size = "network.size"
)

# Compute Gile's SS weights
weights <- compute.weights(rds, weight.type = "Gile's SS", N = N)

# Ensure names
if (is.null(names(weights))) {
  names(weights) <- data$id
}

# Combine and write
result <- data.frame(id = names(weights), weight = as.vector(weights))
colnames(result) <- c("id", "weight")
write.csv(result, file = output_csv, row.names = FALSE)




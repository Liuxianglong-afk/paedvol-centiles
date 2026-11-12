# Dense z-grid of the reference centiles for the paedvol tool (ABIDE + HBN controls, 5-21 years).
#
# Refits each regional GAMLSS on all controls with the families selected in A13 (same fallback rule) and writes,
# for every region x age (5-21 y, step 0.25) x sex, the volume at z = -4.0, -3.9, ..., 4.0.
# A tool can then convert any volume to a z score by interpolation, without R.
# Also writes exact model z scores for a random validation sample (seed 42) so that the interpolation error can be measured.
# Outputs (<outdir>):
#   A31_zgrid.csv.gz            region, family, age, sex, z_-4.0 ... z_4.0 (volumes in cm3)
#   A31_validation_exact_z.csv.gz  participant-level exact z (validation only; not distributed)
# Usage: Rscript scripts/A31_zgrid_centiles.R data/interim/A_joint_norm_input_20261112.csv derivatives/norms_joint/A13_families.csv derivatives/norms_joint [cores]
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(gamlss); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; famf <- args[2]; outd <- args[3]; ncores <- ifelse(length(args) >= 4, as.integer(args[4]), 8)
set.seed(42)
FAMS <- c("BCCGo", "BCTo", "SHASHo")
d <- fread(inp, colClasses = list(character = "sub_n"))
d <- d[age >= 5 & age <= 21]
d[, sex := factor(sex, levels = c("F", "M"))]
regions <- grep("^(G_|S_|V_|C_|CB_)", names(d), value = TRUE)
tc <- d[group == "TC"]
fam_tab <- fread(famf); fam_of <- setNames(fam_tab$family, fam_tab$region)
val <- d[sample(.N, 400)]
ZS <- round(seq(-4, 4, by = 0.1), 1)
grid <- CJ(age = seq(5, 21, by = 0.25), sex = factor(c("F", "M"), levels = c("F", "M")))

fit_one <- function(dat, region, fam) {
  dd <- data.frame(y = dat[[region]], age = dat$age, sex = dat$sex)
  if (fam == "NOlog") dd$y <- log(dd$y)
  f <- if (fam == "NOlog") "NO" else fam
  m <- tryCatch(do.call(gamlss, list(formula = y ~ pb(age) + sex, sigma.formula = ~ pb(age) + sex, family = f, data = dd,
                                     control = gamlss.control(n.cyc = 200, trace = FALSE))), error = function(e) NULL)
  if (!is.null(m)) attr(m, "train") <- dd
  m
}
fit_fallback <- function(dat, region, fam) {
  for (f in unique(c(fam, FAMS, "NOlog"))) { m <- fit_one(dat, region, f); if (!is.null(m)) return(list(m = m, fam = f)) }
  NULL
}
cl <- makeCluster(ncores)
clusterEvalQ(cl, { .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages({ library(gamlss); library(data.table) }) })
clusterExport(cl, c("tc", "val", "grid", "ZS", "fam_of", "FAMS", "fit_one", "fit_fallback"))
res <- parLapply(cl, regions, function(r) {
  ff <- fit_fallback(tc, r, fam_of[[r]]); fam <- ff$fam; m <- ff$m
  base <- ifelse(fam == "NOlog", "NO", fam)
  pa <- predictAll(m, newdata = as.data.frame(grid), data = attr(m, "train"), type = "response")
  qf <- get(paste0("q", base))
  g <- data.frame(region = r, family = fam, age = grid$age, sex = as.character(grid$sex))
  for (z in ZS) {
    argl <- list(p = pnorm(z), mu = pa$mu, sigma = pa$sigma)
    if (!is.null(pa$nu)) argl$nu <- pa$nu
    if (!is.null(pa$tau)) argl$tau <- pa$tau
    v <- do.call(qf, argl); g[[sprintf("z_%.1f", z)]] <- if (fam == "NOlog") exp(v) else v
  }
  pv <- predictAll(m, newdata = data.frame(age = val$age, sex = val$sex), data = attr(m, "train"), type = "response")
  y <- val[[r]]; if (fam == "NOlog") y <- log(y)
  argl <- list(q = y, mu = pv$mu, sigma = pv$sigma)
  if (!is.null(pv$nu)) argl$nu <- pv$nu
  if (!is.null(pv$tau)) argl$tau <- pv$tau
  zx <- qnorm(pmin(pmax(do.call(get(paste0("p", base)), argl), 1e-7), 1 - 1e-7))
  list(grid = g, val = data.frame(site = val$site, sub_n = val$sub_n, age = val$age, sex = as.character(val$sex), region = r, volume = val[[r]], z_exact = zx))
})
stopCluster(cl)
fwrite(rbindlist(lapply(res, `[[`, "grid")), file.path(outd, "A31_zgrid.csv.gz"))
fwrite(rbindlist(lapply(res, `[[`, "val")), file.path(outd, "A31_validation_exact_z.csv.gz"))
cat("DONE regions:", length(regions), "\n")

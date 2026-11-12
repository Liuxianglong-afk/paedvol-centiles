# GAMLSS paediatric brain volume centiles (ABIDE controls, 7-21 years).
#
# Model per region: y ~ family(mu = pb(age) + sex, sigma = pb(age) + sex)
#   Family chosen by BIC among BCCGo / BCTo / SHASHo (once, in all controls; then fixed); NO(log y) if all fail.
#   No site term: a site-agnostic reference that a new site re-centres on local controls (see A03).
# Outputs (derivatives/norms/):
#   A02_families.csv        selected family and BIC per region
#   A02_loso_z.csv.gz       leave-one-site-out z scores of held-out controls (long format; not distributed)
#   A02_disease_z.csv.gz    model-building controls -> z for held-out controls and ASD (not distributed)
#   A02_centiles_full.csv   centile table from all controls (7-21 years, step 0.25, by sex)
# Usage: Rscript scripts/A02_gamlss_norms.R <input csv> <output dir> [cores]
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(gamlss); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; outd <- args[2]; ncores <- ifelse(length(args) >= 3, as.integer(args[3]), 8)
dir.create(outd, recursive = TRUE, showWarnings = FALSE)
set.seed(42)
AGE_MIN <- 7; AGE_MAX <- 21
FAMS <- c("BCCGo", "BCTo", "SHASHo")

d <- fread(inp)
d <- d[age >= AGE_MIN & age <= AGE_MAX]
d[, sex := factor(sex, levels = c("F", "M"))]
regions <- grep("^(G_|S_|V_|C_|CB_)", names(d), value = TRUE)
tc <- d[group == "TC"]
cat("TC:", nrow(tc), " ASD:", nrow(d[group == "ASD"]), " regions:", length(regions), "\n")

fit_one <- function(dat, region, fam) {
  dd <- data.frame(y = dat[[region]], age = dat$age, sex = dat$sex)
  if (fam == "NOlog") dd$y <- log(dd$y)
  f <- if (fam == "NOlog") "NO" else fam

  m <- tryCatch(do.call(gamlss, list(formula = y ~ pb(age) + sex, sigma.formula = ~ pb(age) + sex,
                                     family = f, data = dd,
                                     control = gamlss.control(n.cyc = 200, trace = FALSE))),
                error = function(e) NULL)
  if (!is.null(m)) attr(m, "train") <- dd
  m
}

zscore <- function(m, fam, newdat, region) {
  nd <- data.frame(age = newdat$age, sex = newdat$sex)
  y <- newdat[[region]]
  if (fam == "NOlog") y <- log(y)
  pa <- predictAll(m, newdata = nd, data = attr(m, "train"), type = "response")
  pf <- get(paste0("p", ifelse(fam == "NOlog", "NO", fam)))
  argl <- list(q = y, mu = pa$mu, sigma = pa$sigma)
  if (!is.null(pa$nu)) argl$nu <- pa$nu
  if (!is.null(pa$tau)) argl$tau <- pa$tau
  p <- do.call(pf, argl)
  qnorm(pmin(pmax(p, 1e-7), 1 - 1e-7))
}


cl <- makeCluster(ncores)
clusterEvalQ(cl, { .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages({ library(gamlss); library(data.table) }) })
clusterExport(cl, c("tc", "fit_one", "zscore", "FAMS"))
fam_tab <- rbindlist(parLapply(cl, regions, function(r) {
  bics <- sapply(FAMS, function(f) { m <- fit_one(tc, r, f); if (is.null(m)) NA_real_ else BIC(m) })
  if (all(is.na(bics))) data.table(region = r, family = "NOlog", bic = NA_real_)
  else data.table(region = r, family = FAMS[which.min(bics)], bic = min(bics, na.rm = TRUE))
}))
fwrite(fam_tab, file.path(outd, "A02_families.csv"))
cat("families:", paste(names(table(fam_tab$family)), table(fam_tab$family), collapse = " "), "\n")
fam_of <- setNames(fam_tab$family, fam_tab$region)
clusterExport(cl, "fam_of")


sites <- sort(unique(tc$site))
clusterExport(cl, "sites")
loso <- rbindlist(parLapply(cl, regions, function(r) {
  out <- list()
  for (s in sites) {
    tr <- tc[site != s]; te <- tc[site == s]
    m <- fit_one(tr, r, fam_of[[r]])
    if (is.null(m)) next
    out[[s]] <- data.frame(site = s, sub_n = te$sub_n, cohort = te$cohort, region = r,
                           z = zscore(m, fam_of[[r]], te, r))
  }
  rbindlist(out)
}))
fwrite(loso, file.path(outd, "A02_loso_z.csv.gz"))
cat("loso rows:", nrow(loso), "\n")


trn <- tc[split == "train"]
oth <- d[!(group == "TC" & split == "train")]
clusterExport(cl, c("trn", "oth"))
dz <- rbindlist(parLapply(cl, regions, function(r) {
  m <- fit_one(trn, r, fam_of[[r]])
  if (is.null(m)) return(NULL)
  rbind(data.frame(site = trn$site, sub_n = trn$sub_n, cohort = trn$cohort, group = "TC", split = "train", region = r, z = zscore(m, fam_of[[r]], trn, r)),
        data.frame(site = oth$site, sub_n = oth$sub_n, cohort = oth$cohort, group = oth$group, split = oth$split, region = r, z = zscore(m, fam_of[[r]], oth, r)))
}))
fwrite(dz, file.path(outd, "A02_disease_z.csv.gz"))
cat("disease rows:", nrow(dz), "\n")


grid <- CJ(age = seq(AGE_MIN, AGE_MAX, by = 0.25), sex = factor(c("F", "M"), levels = c("F", "M")))
clusterExport(cl, "grid")
cent <- rbindlist(parLapply(cl, regions, function(r) {
  fam <- fam_of[[r]]; m <- fit_one(tc, r, fam)
  if (is.null(m)) return(NULL)
  pa <- predictAll(m, newdata = as.data.frame(grid), data = attr(m, "train"), type = "response")
  qf <- get(paste0("q", ifelse(fam == "NOlog", "NO", fam)))
  res <- data.frame(region = r, age = grid$age, sex = as.character(grid$sex))
  for (cc in c(0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97)) {
    argl <- list(p = cc, mu = pa$mu, sigma = pa$sigma)
    if (!is.null(pa$nu)) argl$nu <- pa$nu
    if (!is.null(pa$tau)) argl$tau <- pa$tau
    v <- do.call(qf, argl)
    res[[sprintf("P%02d", round(cc * 100))]] <- if (fam == "NOlog") exp(v) else v
  }
  res
}))
fwrite(cent, file.path(outd, "A02_centiles_full.csv"))
stopCluster(cl)
cat("DONE centile rows:", nrow(cent), "\n")

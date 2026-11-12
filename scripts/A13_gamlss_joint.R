# Joint ABIDE + HBN GAMLSS centiles (controls, 5-21 years), leave-one-site-out and between-dataset validation.
#
# Model per region: y ~ family(mu = pb(age) + sex, sigma = pb(age) + sex); family chosen by BIC among BCCGo / BCTo / SHASHo
# in all controls and then fixed; if a fit fails, other families and finally NO(log y) are tried. No site term.
# Outputs (<outdir>):
#   A13_families.csv        family per region
#   A13_loso_z.csv.gz       leave-one-site-out z of held-out controls (participant level; not distributed)
#   A13_xdb_z.csv.gz        ABIDE-only centiles -> HBN controls and HBN-only centiles -> ABIDE controls (not distributed)
#   A13_disease_z.csv.gz    model-building controls -> z for everyone (not distributed)
#   A13_centiles_full.csv   centiles from all controls (5-21 years, step 0.25, by sex)
# Usage: Rscript scripts/A13_gamlss_joint.R data/interim/A_joint_norm_input_20261112.csv derivatives/norms_joint [cores]
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(gamlss); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; outd <- args[2]; ncores <- ifelse(length(args) >= 3, as.integer(args[3]), 8)
dir.create(outd, recursive = TRUE, showWarnings = FALSE)
set.seed(42)
AGE_MIN <- 5; AGE_MAX <- 21
FAMS <- c("BCCGo", "BCTo", "SHASHo")

d <- fread(inp, colClasses = list(character = "sub_n"))
d <- d[age >= AGE_MIN & age <= AGE_MAX]
d[, sex := factor(sex, levels = c("F", "M"))]
regions <- grep("^(G_|S_|V_|C_|CB_)", names(d), value = TRUE)
tc <- d[group == "TC"]
cat("TC:", nrow(tc), "(ABIDE", nrow(tc[db == "ABIDE"]), "/ HBN", nrow(tc[db == "HBN"]), ") all:", nrow(d), " regions:", length(regions), "\n")

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
  qnorm(pmin(pmax(do.call(pf, argl), 1e-7), 1 - 1e-7))
}

fit_fallback <- function(dat, region, fam) {
  for (f in unique(c(fam, FAMS, "NOlog"))) { m <- fit_one(dat, region, f); if (!is.null(m)) return(list(m = m, fam = f)) }
  NULL
}

cl <- makeCluster(ncores)
clusterEvalQ(cl, { .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages({ library(gamlss); library(data.table) }) })
clusterExport(cl, c("tc", "d", "fit_one", "fit_fallback", "zscore", "FAMS"))

t0 <- Sys.time()
fam_tab <- rbindlist(parLapply(cl, regions, function(r) {
  bics <- sapply(FAMS, function(f) { m <- fit_one(tc, r, f); if (is.null(m)) NA_real_ else BIC(m) })
  if (all(is.na(bics))) data.table(region = r, family = "NOlog", bic = NA_real_)
  else data.table(region = r, family = FAMS[which.min(bics)], bic = min(bics, na.rm = TRUE))
}))
fwrite(fam_tab, file.path(outd, "A13_families.csv"))
fam_of <- setNames(fam_tab$family, fam_tab$region)
clusterExport(cl, "fam_of")
cat("families:", paste(names(table(fam_tab$family)), table(fam_tab$family), collapse = " "), " t=", format(Sys.time() - t0), "\n")


sites <- sort(unique(tc$site)); clusterExport(cl, "sites")
loso <- rbindlist(parLapply(cl, regions, function(r) {
  out <- list()
  for (s in sites) {
    tr <- tc[site != s]; te <- tc[site == s]
    ff <- fit_fallback(tr, r, fam_of[[r]]); if (is.null(ff)) next
    out[[s]] <- data.frame(db = te$db, site = s, sub_n = te$sub_n, region = r, fam_used = ff$fam, z = zscore(ff$m, ff$fam, te, r))
  }
  rbindlist(out)
}))
fwrite(loso, file.path(outd, "A13_loso_z.csv.gz"))
cat("loso rows:", nrow(loso), " fallback fits:", sum(loso$fam_used != fam_of[loso$region]), " t=", format(Sys.time() - t0), "\n")


xdb <- rbindlist(parLapply(cl, regions, function(r) {
  out <- list()
  for (dir in c("ABIDE2HBN", "HBN2ABIDE")) {
    src <- if (dir == "ABIDE2HBN") "ABIDE" else "HBN"
    tr <- tc[db == src]; te <- tc[db != src]
    ff <- fit_fallback(tr, r, fam_of[[r]]); if (is.null(ff)) next
    out[[dir]] <- data.frame(dir = dir, db = te$db, site = te$site, sub_n = te$sub_n, age = te$age, region = r,
                             fam_used = ff$fam, z = zscore(ff$m, ff$fam, te, r))
  }
  rbindlist(out)
}))
fwrite(xdb, file.path(outd, "A13_xdb_z.csv.gz"))
cat("xdb rows:", nrow(xdb), " t=", format(Sys.time() - t0), "\n")


trn <- tc[split == "train"]; clusterExport(cl, "trn")
dz <- rbindlist(parLapply(cl, regions, function(r) {
  ff <- fit_fallback(trn, r, fam_of[[r]]); if (is.null(ff)) return(NULL)
  data.frame(db = d$db, site = d$site, sub_n = d$sub_n, group = d$group, split = d$split, region = r,
             fam_used = ff$fam, z = zscore(ff$m, ff$fam, d, r))
}))
fwrite(dz, file.path(outd, "A13_disease_z.csv.gz"))
cat("disease rows:", nrow(dz), " t=", format(Sys.time() - t0), "\n")


grid <- CJ(age = seq(AGE_MIN, AGE_MAX, by = 0.25), sex = factor(c("F", "M"), levels = c("F", "M")))
clusterExport(cl, "grid")
cent <- rbindlist(parLapply(cl, regions, function(r) {
  ff <- fit_fallback(tc, r, fam_of[[r]]); if (is.null(ff)) return(NULL)
  fam <- ff$fam; m <- ff$m
  pa <- predictAll(m, newdata = as.data.frame(grid), data = attr(m, "train"), type = "response")
  qf <- get(paste0("q", ifelse(fam == "NOlog", "NO", fam)))
  res <- data.frame(region = r, family = fam, age = grid$age, sex = as.character(grid$sex))
  for (cc in c(0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97)) {
    argl <- list(p = cc, mu = pa$mu, sigma = pa$sigma)
    if (!is.null(pa$nu)) argl$nu <- pa$nu
    if (!is.null(pa$tau)) argl$tau <- pa$tau
    v <- do.call(qf, argl)
    res[[sprintf("P%02d", round(cc * 100))]] <- if (fam == "NOlog") exp(v) else v
  }
  res
}))
fwrite(cent, file.path(outd, "A13_centiles_full.csv"))
stopCluster(cl)
cat("DONE centile rows:", nrow(cent), " total t=", format(Sys.time() - t0), "\n")

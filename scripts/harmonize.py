"""Site standardisation preserving covariate effects (ComBat-style location/scale correction without empirical Bayes).

Model per feature: x = alpha + C*beta + gamma_site + delta_site*e
- fit(): training data only; OLS with intercept + covariates + site dummies gives alpha, beta (covariate effects)
  and the pooled residual standard deviation sigma.
- transform(): for any site (including sites unseen in training), uses only that site's feature data (never labels)
  to estimate the mean gamma and SD delta of standardised residuals, corrects them and adds back alpha + C*beta.
Diagnostic labels never enter this module. Suggested covariates: age, age squared, sex.
"""
import numpy as np


class SiteStandardizer:
    def __init__(self, min_site_n=5):
        self.min_site_n = min_site_n

    @staticmethod
    def _design(cov, sites, levels):
        dummies = np.column_stack([(sites == s).astype(float) for s in levels[1:]]) if len(levels) > 1 else np.zeros((len(sites), 0))
        return np.column_stack([np.ones(len(sites)), cov, dummies])

    def fit(self, X, cov, sites):
        X = np.asarray(X, float); cov = np.asarray(cov, float); sites = np.asarray(sites)
        levels = np.unique(sites)
        D = self._design(cov, sites, levels)
        coef, *_ = np.linalg.lstsq(D, X, rcond=None)
        k = cov.shape[1]
        site_eff = np.vstack([np.zeros((1, X.shape[1])), coef[1 + k:]])
        w = np.array([(sites == s).mean() for s in levels])
        self.alpha_ = coef[0] + w @ site_eff
        self.beta_ = coef[1:1 + k]
        resid = X - D @ coef
        self.sigma_ = resid.std(axis=0, ddof=D.shape[1] if len(X) > D.shape[1] else 0)
        self.sigma_[self.sigma_ == 0] = 1.0
        return self

    def transform(self, X, cov, sites):
        X = np.asarray(X, float); cov = np.asarray(cov, float); sites = np.asarray(sites)
        fixed = self.alpha_ + cov @ self.beta_
        z = (X - fixed) / self.sigma_
        out = np.empty_like(z)
        for s in np.unique(sites):
            m = sites == s
            g = z[m].mean(axis=0)
            d = z[m].std(axis=0, ddof=1) if m.sum() >= self.min_site_n else np.ones(z.shape[1])
            d[~np.isfinite(d) | (d == 0)] = 1.0
            out[m] = (z[m] - g) / d
        return out * self.sigma_ + fixed

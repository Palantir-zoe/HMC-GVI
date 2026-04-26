###data###
library(MCMCpack)
library(posterior)
library(aplore3)
data("polypharm")
#
n = 500
ni = 7
k = 8
p = 1
#
Poly_id = c(polypharm[,1])
Poly_xx = matrix(data = 1, nrow = n * ni, ncol = k)
Poly_y = numeric(n * ni)
#
Poly_xx[,2] = as.numeric(c(polypharm$gender == 'Male'))
Poly_xx[,3] = as.numeric(c(polypharm$race != 'White'))
Poly_xx[,4] = as.numeric(c(polypharm$age))
Poly_xx[,4] = log(Poly_xx[,4]/10)
Poly_xx[,5] = as.numeric(c(polypharm$mhv4 == '1-5'))
Poly_xx[,6] = as.numeric(c(polypharm$mhv4 == '6-14'))
Poly_xx[,7] = as.numeric(c(polypharm$mhv4 == '> 14'))
Poly_xx[,8] = as.numeric(c(polypharm$inptmhv3 != '0'))
Poly_y = as.numeric(c(polypharm$polypharmacy == 'Yes'))
###log posterior & gradient of log posterior###
#f = 5
sb = 10
sz = 10
library(MCMCpack)

# Target setting
set.seed(1234)
n = 500 * 7
p = 509
logh <- function(x)
{
  uu = x[1:500]
  bt = x[501:508]
  zt = x[509]
  uu1 = rep(uu, each = 7)
  y = 0
  
  e1 = c(Poly_xx %*% bt) + uu1
  y = y + sum(Poly_y * e1 - log(1 + exp(e1))) - 0.5 * exp(-2 * zt) * sum(uu^2)
  y = y - 1/200.0 * sum(bt^2) - 1/200.0 * zt^2 - 500 * zt
  
}
e3 = colSums(Poly_y * Poly_xx)
grad_logh <- function(x)
{
  uu = x[1:500]
  bt = x[501:508]
  zt = x[509]
  uu1 = rep(uu, each = 7)
  
  e2 = exp(c(Poly_xx %*% bt) + uu1) 
  duu = colSums(matrix(Poly_y - e2 / (1 + e2), nrow = 7)) - exp(-2 * zt) * uu
  
  dbt = e3 - 
    colSums(e2 / (1 + e2) * Poly_xx) - 
    bt / 100
  
  dzt = exp(-2 * zt) * sum(uu^2) - zt / 100.0 - 500
  
  y = c(duu, dbt, dzt)
  
}
#burn_in
MALA_burnin <- function(burn_in, epsilon, S) {
  d <- 509
  out <- matrix(0, burn_in, d) # Initialize an empty matrix to store the values
  thetax <- mu # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  lgrad <- c(S %*% grad_logh(thetax)) # Compute the gradient
  logp <- logh(thetax)
  sigma2 <- epsilon^2 / d^(1 / 3)
  sigma <- sqrt(sigma2)
  # Starting the Gibbs sampling
  for (r in 1:burn_in) {
    thetax_new <- thetax + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(d)))
    logpnew <- logh(thetax_new)
    lgrad_new <- c(S %*% grad_logh(thetax_new))
    diffold <- thetax - thetax_new - sigma2 / 2 * lgrad_new
    diffnew <- thetax_new - thetax - sigma2 / 2 * lgrad
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      thetax <- thetax_new # Accept the value
    }
    # Store the values after the burn-in period
    out[r, ] <- thetax
  }
  out
}
set.seed(123)
epsilon <- 0.15 # After some trial ad error
burn_in=70000
R=1000000
# Running the MCMC
system.time(fit_MALA_burnin <- as.mcmc(MALA_burnin(burn_in = burn_in, epsilon, S = diag(509)))) # Convert the matrix into a "coda" object
# 用户   系统   流逝 
# 57.07   3.70 100.10 

# Diagnostic
summary(effectiveSize(fit_MALA_burnin)) # Effective sample size
#summary(R / effectiveSize(fit_MALA_burnin)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MALA_burnin)) # Acceptance rate
#burnin_estimate
S_burnin=cov(fit_MALA_burnin)
mu = mean(fit_MALA_burnin)
burn_in=10000
#tune epsilon in MALA via burn_in covariance estimate
epsilon <- 0.522 #after some trial and error
fit_MALA_burnin_tune <- as.mcmc(MALA_burnin(burn_in = burn_in, epsilon, S = S_burnin)) # Convert the matrix into a "coda" object
summary(effectiveSize(fit_MALA_burnin_tune)) # Effective sample size
#summary(burn_in / effectiveSize(fit_MALA_burnin_tune)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MALA_burnin_tune)) # Acceptance rate



#
MALA <- function(R, burn_in, epsilon, S) {
  d <- 509
  out <- matrix(0, R, d) # Initialize an empty matrix to store the values
  thetax <- mu # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  
  lgrad <- c(S %*% grad_logh(thetax)) # Compute the gradient
  logp <- logh(thetax)
  
  sigma2 <- epsilon^2 / d^(1 / 3)
  sigma <- sqrt(sigma2)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    thetax_new <- thetax + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(d)))
    
    logpnew <- logh(thetax_new)
    lgrad_new <- c(S %*% grad_logh(thetax_new))
    
    diffold <- thetax - thetax_new - sigma2 / 2 * lgrad_new
    diffnew <- thetax_new - thetax - sigma2 / 2 * lgrad
    
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      thetax <- thetax_new # Accept the value
    }
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- thetax
    }
  }
  out
}
set.seed(123)
epsilon <- 0.522 # After some trial ad error
# Running the MCMC
R=1000000
burn_in=10000
system.time(fit_MCMC_MALA_byburnin <- as.mcmc(MALA(R = R, burn_in = burn_in, epsilon, S = S_burnin))) # Convert the matrix into a "coda" object
# 用户    系统    流逝 
# 1010.38   90.14 2283.83 
# Diagnostic
summary(effectiveSize(fit_MCMC_MALA_byburnin)) # Effective sample size
summary(R / effectiveSize(fit_MCMC_MALA_byburnin)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_MALA_byburnin)) # Acceptance rate

#vi
#tune epsilon in MALA via GVI covariance estimate
epsilon <- 0.07 #after some trial and error
fit_MALA_burnin_tune <- as.mcmc(MALA_burnin(burn_in = burn_in, epsilon, S = S_gvb)) # Convert the matrix into a "coda" object
summary(effectiveSize(fit_MALA_burnin_tune)) # Effective sample size
#summary(burn_in / effectiveSize(fit_MALA_burnin_tune)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MALA_burnin_tune)) # Acceptance rate
#
set.seed(123)
epsilon <- 1.354  # After some trial ad error
# Running the MCMC
R=1000000
burn_in=10000
system.time(fit_MALAbyvi <- as.mcmc(MALA(R = R, burn_in = burn_in, epsilon=epsilon, S = Sigma))) # Convert the matrix into a "coda" object
# Diagnostic
summary(effectiveSize(fit_MALAbyvi)) # Effective sample size
summary(1 - rejectionRate(fit_MALAbyvi))
ess <- rep(0,509)
for(i in 1:509){
  ess[i] = ess_basic(fit_MALAbyvi[,i])
}
set.seed(66)
p=100
mu=rep(0,p)
mat = matrix(0.8,nrow=p,ncol=p)
diag(mat) = rgamma(p,shape=2,scale=3)
cov=mat
inv_cov = solve(cov)
log_tar = function(theta, cov, mu) 
{
  inv_cov = solve(cov)
  return(-0.5 *t(theta - mu)%*%inv_cov%*%(theta - mu))
}
RMH <- function(burn_in,mu,cov, S) {
  p <- ncol(X)
  out <- matrix(0, burn_in, p) # Initialize an empty matrix to store the values
  theta <- rep(0, p) # Initial values
  logp <- log_tar(theta, cov, mu)
  
  # Eigen-decomposition
  eig <- eigen(S, symmetric = TRUE)
  A1 <- t(eig$vectors) * sqrt(eig$values)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in)) {
    theta_new <- theta + c(matrix(rnorm(p), 1, p) %*% A1)
    logp_new <- logpost(theta_new, cov,mu)
    alpha <- min(1, exp(logp_new - logp))
    if (runif(1) < alpha) {
      logp <- logp_new
      theta <- theta_new # Accept the value
    }
    # Store the values after the burn-in period
    out[r, ] <- theta
  }
  out
}
library(coda)
burn_in <- 10000 # Burn-in period
set.seed(123)
# Covariance matrix of the proposal
S <- diag(1e-3, ncol(X))
# Running the MCMC
start.time <- Sys.time()
system.time(fit_MCMC <- as.mcmc(RMH(burn_in, y, X, S))) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(end.time - start.time)
Sigma=cov(fit_MCMC)
S=2.38^2*Sigma/p
# 用户 系统 流逝 
# 0.18 0.01 0.38 

# R represent the number of samples
# burn_in is the number of discarded samples
# S is the covariance matrix of the multivariate Gaussian proposal
RMH <- function(R, burn_in, mu,cov, S) {
  p <- ncol(X)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  theta <- rep(0, p) # Initial values
  logp <- log_tar(theta, cov, mu)
  
  # Eigen-decomposition
  eig <- eigen(S, symmetric = TRUE)
  A1 <- t(eig$vectors) * sqrt(eig$values)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    theta_new <- theta + c(matrix(rnorm(p), 1, p) %*% A1)
    logp_new <- log_tar(theta, cov, mu)
    alpha <- min(1, exp(logp_new - logp))
    if (runif(1) < alpha) {
      logp <- logp_new
      theta <- theta_new # Accept the value
    }
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- theta
    }
  }
  out
}
R=1000000
burn_in=10000
# Running the MCMC
system.time(fit_MCMC <- as.mcmc(RMH(burn_in,mu,cov, S)))# Convert the matrix into a "coda" object
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation



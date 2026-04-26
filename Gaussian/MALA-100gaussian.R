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
#gradient
G_tar = function(theta, cov, mu) 
{
  return(-inv_cov %*% (theta - mu)) # transpose the first matrix)
}
MALA_burnin <- function( burn_in, mu, cov, epsilon, S) {
  p <- ncol(cov)
  out <- matrix(0, burn_in, p) # Initialize an empty matrix to store the values
  theta <- rep(0, p) # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  
  lgrad <- c(S %*% G_tar(theta, cov, mu)) # Compute the gradient
  logp <- log_tar(theta, cov, mu)
  
  sigma2 <- epsilon^2 / p^(1 / 3)
  sigma <- sqrt(sigma2)
  
  # Starting the Gibbs sampling
  for (r in 1:burn_in) {
    theta_new <- theta + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(p)))
    
    logpnew <- log_tar(theta_new, cov, mu)
    lgrad_new <- c(S %*% G_tar(theta_new, cov, mu))
    
    diffold <- theta - theta_new - sigma2 / 2 * lgrad_new
    diffnew <- theta_new -theta - sigma2 / 2 * lgrad
    
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      theta <- theta_new # Accept the value
    }
    # Store the values after the burn-in period
    out[r, ] <- theta
  }
  out
}
set.seed(123)
epsilon <- 1.53 # After some trial ad error
burn_in=10000
# Running the MCMC
start.time <- Sys.time()
fit_MCMC_burnin <- as.mcmc(MALA_burnin( burn_in = burn_in, mu, cov, epsilon, S = diag(p))) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
samples_burnin = fit_MCMC_burnin
S=cov(samples_burnin)
# Diagnostic
summary(effectiveSize(fit_MCMC_burnin)) # Effective sample size
summary(R / effectiveSize(fit_MCMC_burnin)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_burnin)) # Acceptance rate
# R represent the number of samples
# burn_in is the number of discarded samples
# epsilon, S are tuning parameter
MALA <- function(R, burn_in, mu, cov, epsilon, S) {
  p <- ncol(cov)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  theta <- rep(0, p) # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  
  lgrad <- c(S %*% G_tar(theta, cov, mu)) # Compute the gradient
  logp <- log_tar(theta, cov, mu)
  
  sigma2 <- epsilon^2 / p^(1 / 3)
  sigma <- sqrt(sigma2)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    theta_new <- theta + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(p)))
    
    logpnew <- log_tar(theta_new, cov, mu)
    lgrad_new <- c(S %*% G_tar(theta_new, cov, mu))
    
    diffold <- theta - theta_new - sigma2 / 2 * lgrad_new
    diffnew <- theta_new -theta - sigma2 / 2 * lgrad
    
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      theta <- theta_new # Accept the value
    }
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- theta
    }
  }
  out
}
set.seed(123)
epsilon <- 1.515 # After some trial ad error
R=1000000
burn_in=10000
# Running the MCMC
start.time <- Sys.time()
fit_MCMC <- as.mcmc(MALA(R = R, burn_in = burn_in, mu, cov, epsilon, S )) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(R / effectiveSize(fit_MCMC)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
samples=fit_MCMC
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
#mala vi

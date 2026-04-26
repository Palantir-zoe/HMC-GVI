mu_gvi=read.csv()
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
HMC <- function(R, burn_in, epsilon, S, L) {
  p <- 509
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  beta <- mu_gvi # Initial values
  logp <- logh(beta) # Initial log-posterior
  S1 <- solve(S)
  A1 <- chol(S1)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    P <- c(crossprod(A1, rnorm(p))) # Auxiliary variables
    logK <- c(P %*% S %*% P / 2) # Kinetic energy at the beginning of the trajectory
    
    # Make a half step for momentum at the beginning
    beta_new <- beta
    Pnew <- P + epsilon * grad_logh(beta_new) / 2
    
    # Alternate full steps for position and momentum
    for (l in 1:L) {
      # Make a full step for the position
      beta_new <- beta_new + epsilon * c(S %*% Pnew)
      # Make a full step for the momentum, except at end of trajectory
      if (l != L) Pnew <- Pnew + epsilon * grad_logh(beta_new)
    }
    # Make a half step for momentum at the end.
    Pnew <- Pnew + epsilon * grad_logh(beta_new) / 2
    
    # Negate momentum at end of trajectory to make the proposal symmetric
    Pnew <- - Pnew
    
    # Evaluate potential and kinetic energies at the end of trajectory
    logpnew <- logh(beta_new)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    if (runif(1) < exp(logpnew - logp + logK - logKnew)) {
      logp <- logpnew
      beta <- beta_new # Accept the value
    }
    
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- beta
    }
  }
  out
}
set.seed(123)
epsilon <- 0.185 # After some trial ad error
L <- 10
# Covariance matrix is selected via laplace approximation
S=Sigma_gvi
R=1000000
burn_in=10000
# Running the MCMC
system.time(fit_MCMC <- as.mcmc(HMC(R = R, burn_in = burn_in, epsilon, S, L))) # Convert the matrix into a "coda" object
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(1-rejectionRate(fit_MCMC)) 
samples=fit_MCMC
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation

calculate_rho1 <- function(samples) {
  n <- length(samples)
  mean_samples <- mean(samples)
  numerator <- sum((samples[-1] - mean_samples) * (samples[-n] - mean_samples))
  denominator <- sum((samples - mean_samples)^2)
  rho1 <- numerator / denominator
  return(rho1)
}

#rho1
rho1 <- calculate_rho1(fit_HMC_byburnin)


#burn in
HMC_burnin <- function(burn_in, epsilon, S, L) {
  p <- 509
  out <- matrix(0, burn_in, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  logp <- logh(beta) # Initial log-posterior
  S1 <- solve(S)
  A1 <- chol(S1)
  
  # Starting the Gibbs sampling
  for (r in 1:burn_in){
    P <- c(crossprod(A1, rnorm(p))) # Auxiliary variables
    logK <- c(P %*% S %*% P / 2) # Kinetic energy at the beginning of the trajectory
    
    # Make a half step for momentum at the beginning
    beta_new <- beta
    Pnew <- P + epsilon * grad_logh(beta_new) / 2
    
    # Alternate full steps for position and momentum
    for (l in 1:L) {
      # Make a full step for the position
      beta_new <- beta_new + epsilon * c(S %*% Pnew)
      # Make a full step for the momentum, except at end of trajectory
      if (l != L) Pnew <- Pnew + epsilon * grad_logh(beta_new)
    }
    # Make a half step for momentum at the end.
    Pnew <- Pnew + epsilon * grad_logh(beta_new) / 2
    
    # Negate momentum at end of trajectory to make the proposal symmetric
    Pnew <- - Pnew
    
    # Evaluate potential and kinetic energies at the end of trajectory
    logpnew <- logh(beta_new)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    if (runif(1) < exp(logpnew - logp + logK - logKnew)) {
      logp <- logpnew
      beta <- beta_new # Accept the value
    }
    
    # Store the values after the burn-in period
    out[r, ] <- thetax <- beta
  }
  out
}
set.seed(123)
epsilon <- 0.03# After some trial ad error0.035
L <- 40
S=diag(509)
burn_in=10000
start.time <- Sys.time()
fit_HMC_burnin <- as.mcmc(HMC_burnin(burn_in ,epsilon, S, L)) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
# Diagnostic
summary(effectiveSize(fit_HMC_burnin)) # Effective sample size
rho1 <- calculate_rho1(fit_HMC_burnin)
rho1 
#burnin_turn
S_burnin=cov(fit_HMC_burnin)
set.seed(123)
epsilon <- 0.08# After some trial ad error
L <- 20
S=S_burnin
burn_in=10000
start.time <- Sys.time()
fit_HMC_burnin <- as.mcmc(HMC_burnin(burn_in ,epsilon, S, L)) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
# Diagnostic
summary(effectiveSize(fit_HMC_burnin)) # Effective sample size
rho1 <- calculate_rho1(fit_HMC_burnin)
rho1 

#hmc_burnin
set.seed(123)
epsilon <- 0.05 # After some trial ad error
L <- 20
# Covariance matrix is selected via laplace approximation
S=S_burnin
R=1000000
burn_in=10000
# Running the MCMC
start.time <- Sys.time()
fit_HMC_byburnin <- as.mcmc(HMC(R = R, burn_in = burn_in, epsilon, S=S_burnin, L)) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
ess <- rep(0,509)
for(i in 1:509){
  ess[i] = ess_basic(fit_HMC_byburnin[,i])
}
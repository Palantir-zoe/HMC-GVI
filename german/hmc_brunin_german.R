german <- read.csv("C:/Users/Angel/Desktop/paper_2/HMC-GVI/german/german_num.csv", header = TRUE, sep = ",")
y <- as.numeric(german$Creditability==1)# Binary outcome
X <- model.matrix(Creditability ~ . - 1, data = german) # Design matrix
X <- cbind(1, scale(X)) # Standardize the design matrix, add the intercept
# Logit model
fit_logit <- glm(y ~ X - 1, family = binomial(link = "logit"), data = german)
# Loglikelihood of a logistic regression model
loglik <- function(beta, y, X) {
  eta <- c(X %*% beta)
  sum(y * eta - log(1 + exp(eta)))
}
# Logposterior
logpost <- function(beta, y, X) {
  loglik(beta, y, X) + sum(dnorm(beta, 0, 10, log = T))
}
lgradient <- function(thetax, y, X) {
  probs <- plogis(c(X %*% thetax))
  loglik_gr <- c(crossprod(X, y - probs))
  prior_gr <- -thetax / 100
  loglik_gr + prior_gr
}
HMC_burnin <- function( burn_in, y, X, epsilon=0.08, S, L=32) {
  p <- ncol(X)
  out_burnin <- matrix(0, burn_in, p)
  beta <- rep(0, p) # Initial values
  logp <- logpost(beta, y, X) # Initial log-posterior
  S1 <- solve(S)
  A1 <- chol(S1)
  for (r in 1:burn_in) {
    P <- c(crossprod(A1, rnorm(p))) # Auxiliary variables
    logK <- c(P %*% S %*% P / 2) # Kinetic energy at the beginning of the trajectory
    # Make a half step for momentum at the beginning
    beta_new <- beta
    Pnew <- P + epsilon * lgradient(beta_new, y, X) / 2
    # Alternate full steps for position and momentum
    for (l in 1:L) {
      # Make a full step for the position
      beta_new <- beta_new + epsilon * c(S %*% Pnew)
      # Make a full step for the momentum, except at end of trajectory
      if (l != L) Pnew <- Pnew + epsilon * lgradient(beta_new, y, X)
    }
    # Make a half step for momentum at the end.
    Pnew <- Pnew + epsilon * lgradient(beta_new, y, X) / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    Pnew <- - Pnew
    # Evaluate potential and kinetic energies at the end of trajectory
    logpnew <- logpost(beta_new, y, X)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    if (runif(1) < exp(logpnew - logp + logK - logKnew)) {
      logp <- logpnew
      beta <- beta_new # Accept the value
    }
    # Store the values after the burn-in period
    out_burnin[r, ] <- beta
  }
  return(out_burnin)
}
set.seed(123)
# Covariance matrix 
S <- diag(ncol(X))
burn_in = 10000
epsilon<-0.03
L=30
# Running the MCMC
system.time(fit_MCMC_5 <- as.mcmc(HMC_burnin(burn_in = burn_in, y, X, epsilon, S, L))) # Convert the matrix into a "coda" object
# 用户  系统  流逝 
# 8.94  0.03 22.40 

# Diagnostic
summary(effectiveSize(fit_MCMC_5)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC_5)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_5)) # Acceptance rate
acceptance_rate <-1 - rejectionRate(fit_MCMC_5)
p_jump <- mean(acceptance_rate)
acf_result <- acf(fit_MCMC_5, lag.max = 1, plot = FALSE)
first_order_autocorrelation <- acf_result$acf[2]
first_order_autocorrelation

# > summary(effectiveSize(fit_MCMC_5)) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 209.4   836.5  1709.1  3286.0  3055.8 14743.9 
# > summary(burn_in / effectiveSize(fit_MCMC_5)) # Integrated autocorrelation time
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.6783  3.2725  5.8510 10.4711 11.9548 47.7625 
# > summary(1 - rejectionRate(fit_MCMC_5)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.9441  0.9441  0.9441  0.9441  0.9441  0.9441 
# > acceptance_rate <-1 - rejectionRate(fit_MCMC_5)
# > p_jump <- mean(acceptance_rate)
# > acf_result <- acf(fit_MCMC_5, lag.max = 1, plot = FALSE)
# > first_order_autocorrelation <- acf_result$acf[2]
# > first_order_autocorrelation
# [1] 0.03212355

HMC <- function(R, burn_in, y, X, epsilon, S, L) {
  p <- ncol(X)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  logp <- logpost(beta, y, X) # Initial log-posterior
  S1 <- solve(S)
  A1 <- chol(S1)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    P <- c(crossprod(A1, rnorm(p))) # Auxiliary variables
    logK <- c(P %*% S %*% P / 2) # Kinetic energy at the beginning of the trajectory
    
    # Make a half step for momentum at the beginning
    beta_new <- beta
    Pnew <- P + epsilon * lgradient(beta_new, y, X) / 2
    
    # Alternate full steps for position and momentum
    for (l in 1:L) {
      # Make a full step for the position
      beta_new <- beta_new + epsilon * c(S %*% Pnew)
      # Make a full step for the momentum, except at the end of the trajectory
      if (l != L) Pnew <- Pnew + epsilon * lgradient(beta_new, y, X)
    }
    # Make a half step for momentum at the end.
    Pnew <- Pnew + epsilon * lgradient(beta_new, y, X) / 2
    
    # Negate momentum at the end of the trajectory to make the proposal symmetric
    Pnew <- - Pnew
    
    # Evaluate potential and kinetic energies at the end of the trajectory
    logpnew <- logpost(beta_new, y, X)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    
    # Accept or reject the state at the end of the trajectory, returning either
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
epsilon<-0.155
L=10
R=1000000
burn_in=10000
S=cov(fit_MCMC_5)
system.time(fit_MCMC_6 <- as.mcmc(HMC(R=R,burn_in = burn_in, y, X, epsilon, S, L))) # Convert the matrix into a "coda" object
# 用户    系统    流逝 
# 843.72    1.75 1972.75 
summary(effectiveSize(fit_MCMC_6)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC_6)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_6)) # Acceptance rate
acceptance_rate <-1 - rejectionRate(fit_MCMC_6)
p_jump <- mean(acceptance_rate)
# > summary(effectiveSize(fit_MCMC_6)) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 848121  921327  946681  945789  972849 1095029 
# > summary(burn_in / effectiveSize(fit_MCMC_6)) # Integrated autocorrelation time
# Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.009132 0.010279 0.010563 0.010601 0.010854 0.011791 
# > summary(1 - rejectionRate(fit_MCMC_6)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.9876  0.9876  0.9876  0.9876  0.9876  0.9876 
acf_result <- acf(fit_MCMC_6, lag.max = 1, plot = FALSE)
first_order_autocorrelation <- acf_result$acf[2]
first_order_autocorrelation
# > first_order_autocorrelation
# [1] 0.003138858
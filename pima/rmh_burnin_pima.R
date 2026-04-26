library(coda)
Pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
y <- as.numeric(Pima$type == "Yes") # Binary outcome
X <- model.matrix(type ~ . - 1, data = Pima) # Design matrix
X <- cbind(1, scale(X)) # Standardize the design matrix, add the intercept
fit_logit <- glm(type ~ X - 1, family = binomial(link = "logit"), data = Pima)
# Loglikelihood of a logistic regression model
loglik <- function(beta, y, X) {
  eta <- c(X %*% beta)
  sum(y * eta - log(1 + exp(eta)))
}
# Logposterior
logpost <- function(beta, y, X) {
  loglik(beta, y, X) + sum(dnorm(beta, 0, 10, log = T))
}
# R represent the number of samples
# burn_in is the number of discarded samples
# S is the covariance matrix of the multivariate Gaussian proposal
RMH <- function( burn_in, y, X, S) {
  p <- ncol(X)
  out <- matrix(0, burn_in, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  logp <- logpost(beta, y, X)
  
  # Eigen-decomposition
  eig <- eigen(S, symmetric = TRUE)
  A1 <- t(eig$vectors) * sqrt(eig$values)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in)) {
    beta_new <- beta + c(matrix(rnorm(p), 1, p) %*% A1)
    logp_new <- logpost(beta_new, y, X)
    alpha <- min(1, exp(logp_new - logp))
    if (runif(1) < alpha) {
      logp <- logp_new
      beta <- beta_new # Accept the value
    }
    # Store the values after the burn-in period
    out[r, ] <- beta
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
RMH <- function(R, burn_in, y, X, S) {
  p <- ncol(X)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  logp <- logpost(beta, y, X)
  
  # Eigen-decomposition
  eig <- eigen(S, symmetric = TRUE)
  A1 <- t(eig$vectors) * sqrt(eig$values)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    beta_new <- beta + c(matrix(rnorm(p), 1, p) %*% A1)
    logp_new <- logpost(beta_new, y, X)
    alpha <- min(1, exp(logp_new - logp))
    if (runif(1) < alpha) {
      logp <- logp_new
      beta <- beta_new # Accept the value
    }
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- beta
    }
  }
  out
}
R=1000000
burn_in=10000
# Running the MCMC
system.time(fit_MCMC <- as.mcmc(RMH(R, burn_in, y, X, S)))# Convert the matrix into a "coda" object
# 用户  系统  流逝 
# 14.43  0.31 34.88 
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation

# > # Diagnostic
#   > summary(effectiveSize(fit_MCMC)) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 32759   35430   35918   38019   38159   48311
# > summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 0.2392  0.2392  0.2392  0.2392  0.2392  0.2392
# > autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation
# [1] 0.909869

#trace_plot and acf_plot
beta1_chain <- fit_MCMC[1:10000, 1]
# Plot Trace and ACF
par(mfrow=c(1,2))  # Set up the plotting area to have 1 row and 2 columns
traceplot(as.mcmc(beta1_chain), main="", xlab = "Iteration", ylab = "", ylim=c(-1.6,-0.5))
acf(beta1_chain, main = "",xlim=c(0,60),ylim=c(0,1))
mtext("RMH", side=3, line=-2, outer=TRUE, cex=1.5)
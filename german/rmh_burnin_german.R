german <- read.csv("C:/Users/14570/Desktop/paper/HMC-GVI/german/german_num.csv", header = TRUE, sep = ",")
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
# 0.20 0.00 0.71 

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
system.time(fit_MCMC <- as.mcmc(RMH(R, burn_in, y, X, S))) # Convert the matrix into a "coda" object
# 用户  系统  流逝 
# 24.05  0.07 66.06
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
# > # Diagnostic
#   > summary(effectiveSize(fit_MCMC)) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 9954   11002   11499   12184   12534   18435 
# > summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.2379  0.2379  0.2379  0.2379  0.2379  0.2379 
# > autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation
# [1] 0.959493
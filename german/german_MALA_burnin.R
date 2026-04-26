german <- read.csv("german_num.csv", header = TRUE, sep = ",")
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
MALA <- function(burn_in, y, X, epsilon, S) {
  p <- ncol(X)
  out <- matrix(0, burn_in, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  
  lgrad <- c(S %*% lgradient(beta, y, X)) # Compute the gradient
  logp <- logpost(beta, y, X)
  
  sigma2 <- epsilon^2 / p^(1 / 3)
  sigma <- sqrt(sigma2)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in)) {
    beta_new <- beta + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(p)))
    
    logpnew <- logpost(beta_new, y, X)
    lgrad_new <- c(S %*% lgradient(beta_new, y, X))
    
    diffold <- beta - beta_new - sigma2 / 2 * lgrad_new
    diffnew <- beta_new - beta - sigma2 / 2 * lgrad
    
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      beta <- beta_new # Accept the value
    }
    # Store the values after the burn-in period
    out[r, ] <- beta
  }
  out
}
burn_in=10000
set.seed(123)
epsilon <- 0.12 # After some trial ad error
# Running the MCMC
start.time <- Sys.time()
fit_MCMC <- as.mcmc(MALA( burn_in = burn_in, y, X, epsilon, S = diag(ncol(X)))) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate

# R represent the number of samples
# burn_in is the number of discarded samples
# epsilon, S are tuning parameter
MALA <- function(R, burn_in, y, X, epsilon, S) {
  p <- ncol(X)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  A <- chol(S) # Cholesky of S
  S1 <- solve(S) # Inverse of S
  lgrad <- c(S %*% lgradient(beta, y, X)) # Compute the gradient
  logp <- logpost(beta, y, X)
  
  sigma2 <- epsilon^2 / p^(1 / 3)
  sigma <- sqrt(sigma2)
  
  # Starting the Gibbs sampling
  for (r in 1:(burn_in + R)) {
    beta_new <- beta + sigma2 / 2 * lgrad + sigma * c(crossprod(A, rnorm(p)))
    
    logpnew <- logpost(beta_new, y, X)
    lgrad_new <- c(S %*% lgradient(beta_new, y, X))
    
    diffold <- beta - beta_new - sigma2 / 2 * lgrad_new
    diffnew <- beta_new - beta - sigma2 / 2 * lgrad
    
    qold <- -diffold %*% S1 %*% diffold / (2 * sigma2)
    qnew <- -diffnew %*% S1 %*% diffnew / (2 * sigma2)
    
    alpha <- min(1, exp(logpnew - logp + qold - qnew))
    if (runif(1) < alpha) {
      logp <- logpnew
      lgrad <- lgrad_new
      beta <- beta_new # Accept the value
    }
    # Store the values after the burn-in period
    if (r > burn_in) {
      out[r - burn_in, ] <- beta
    }
  }
  out
}
library(coda)
R <- 1000000
burn_in <- 10000
set.seed(123)
epsilon <- 1.68# After some trial ad error
# Running the MCMC
S=cov(fit_MCMC)
start.time <- Sys.time()
fit_MCMC1 <- as.mcmc(MALA(R = R, burn_in = burn_in, y, X, epsilon, S)) # Convert the matrix into a "coda" object
end.time <- Sys.time()
time_in_sec <- as.numeric(difftime(end.time, start.time, units = "secs"))
# Diagnostic
summary(effectiveSize(fit_MCMC1)) # Effective sample size
summary(R / effectiveSize(fit_MCMC1)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC1)) # Acceptance rate
autocorrelation <- acf(fit_MCMC, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation


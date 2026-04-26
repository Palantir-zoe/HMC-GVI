library(coda)
library(MCMCpack)
Pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
write.csv(Pima, "Pima_dataset.csv", row.names = TRUE)
y <- as.numeric(Pima$type == "Yes") # Binary outcome
X <- model.matrix(type ~ . - 1, data = Pima) # Design matrix
X <- cbind(1, scale(X)) # Standardize the design matrix, add the intercept
# Logit model
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
epsilon <- 0.175 # After some trial ad error
# Running the MCMC
system.time(fit_MCMC <- as.mcmc(MALA( burn_in = burn_in, y, X, epsilon, S = diag(ncol(X))))) # Convert the matrix into a "coda" object
# Diagnostic
summary(effectiveSize(fit_MCMC)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC)) # Acceptance rate
#用户 系统 流逝 
#0.38 0.00 0.69 
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
system.time(fit_MCMC1 <- as.mcmc(MALA(R = R, burn_in = burn_in, y, X, epsilon, S))) # Convert the matrix into a "coda" object
#用户  系统  流逝 
#23.56  0.72 63.00
# Diagnostic
summary(effectiveSize(fit_MCMC1)) # Effective sample size
summary(R / effectiveSize(fit_MCMC1)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC1)) # Acceptance rate
autocorrelation <- acf(fit_MCMC1, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
#> # Diagnostic
#  > summary(effectiveSize(fit_MCMC1)) # Effective sample size
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#292832  296314  301047  299548  302389  304368 
#> summary(R / effectiveSize(fit_MCMC1)) # Integrated autocorrelation time
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#3.285   3.307   3.322   3.339   3.375   3.415 
#> summary(1 - rejectionRate(fit_MCMC1)) # Acceptance rate
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.5546  0.5546  0.5546  0.5546  0.5546  0.5546 
# > autocorrelation <- acf(fit_MCMC1, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation
# [1] 0.5028918


#trace_plot and acf_plot
beta1_chain <- fit_MCMC1[1:10000, 1]
# Plot Trace and ACF
par(mfrow=c(1,2))  # Set up the plotting area to have 1 row and 2 columns
traceplot(as.mcmc(beta1_chain), main="", xlab = "Iteration", ylab = "", ylim=c(-1.6,-0.5))
acf(beta1_chain, main = "",xlim=c(0,60),ylim=c(0,1))
mtext("AM", side=3, line=-2, outer=TRUE, cex=1.5)
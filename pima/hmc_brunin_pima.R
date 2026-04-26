Pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
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
# Gradient of the logposterior
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
epsilon<-0.10
L=25
# Running the MCMC
system.time(fit_MCMC_5 <- as.mcmc(HMC_burnin(burn_in = burn_in, y, X, epsilon, S, L))) # Convert the matrix into a "coda" object
#用户 系统 流逝 
#2.33 0.01 7.67
# Diagnostic
summary(effectiveSize(fit_MCMC_5)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC_5)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_5)) # Acceptance rate
acceptance_rate <-1 - rejectionRate(fit_MCMC_5)
p_jump <- mean(acceptance_rate)
#> # Diagnostic
 # > summary(effectiveSize(fit_MCMC_5)) # Effective sample size
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#154     560    1066    1750    3267    4008 
#> summary(burn_in / effectiveSize(fit_MCMC_5)) # Integrated autocorrelation time
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#2.495   3.076   9.865  19.257  22.665  64.920 
#> summary(1 - rejectionRate(fit_MCMC_5)) # Acceptance rate
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.7702  0.7702  0.7702  0.7702  0.7702  0.7702
acf_result <- acf(fit_MCMC_5, lag.max = 1, plot = FALSE)
first_order_autocorrelation <- acf_result$acf[2]
first_order_autocorrelation
#[1] -0.07315369
HMC <- function(R, burn_in, y, X, epsilon, S, L = 10) {
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
epsilon<-0.15
L=10
S=cov(fit_MCMC_5)
system.time(fit_MCMC_6 <- as.mcmc(HMC(R=R,burn_in = burn_in, y, X, epsilon, S, L))) # Convert the matrix into a "coda" object
#用户   系统   流逝 
#117.66   0.25 344.43 
summary(effectiveSize(fit_MCMC_6)) # Effective sample size
summary(burn_in / effectiveSize(fit_MCMC_6)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_6)) # Acceptance rate
acceptance_rate <-1 - rejectionRate(fit_MCMC_6)
p_jump <- mean(acceptance_rate)
#> summary(effectiveSize(fit_MCMC_6)) # Effective sample size
#Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#7576816  7933878  8751795  9189203 10148428 12296358 
#> summary(burn_in / effectiveSize(fit_MCMC_6)) # Integrated autocorrelation time
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0008132 0.0009854 0.0011427 0.0011144 0.0012604 0.0013198 
#> summary(1 - rejectionRate(fit_MCMC_6)) # Acceptance rate
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.9902  0.9902  0.9902  0.9902  0.9902  0.9902 
#> acceptance_rate <-1 - rejectionRate(fit_MCMC_6)
#> p_jump <- mean(acceptance_rate)
#> acceptance_rate
#var1    var2    var3    var4    var5    var6    var7    var8 
#0.99022 0.99022 0.99022 0.99022 0.99022 0.99022 0.99022 0.99022 
acf_result <- acf(fit_MCMC_6, lag.max = 1, plot = FALSE)
first_order_autocorrelation <- acf_result$acf[2]
first_order_autocorrelation
#> first_order_autocorrelation
#[1] -0.01794569
#trace_plot and acf_plot
beta1_chain <- fit_MCMC_6[1:10000, 1]
# Plot Trace and ACF
par(mfrow=c(1,2))  # Set up the plotting area to have 1 row and 2 columns
traceplot(as.mcmc(beta1_chain), main="", xlab = "Iteration", ylab = "", ylim=c(-1.6,-0.5))
acf(beta1_chain, xlim=c(0,60),ylim=c(0,1),main = "")
mtext("HMC", side=3, line=-2, outer=TRUE, cex=1.5)

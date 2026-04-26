Pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
y <- as.numeric(Pima$type == "Yes") # Binary outcome
X <- model.matrix(type ~ . - 1, data = Pima) # Design matrix
X <- cbind(1, scale(X)) # Standardize the design matrix, add the intercept
#Adaptive Metropolis-Hastings
RMH_Adaptive <- function(R, burn_in, y, X) {
  p <- ncol(X)
  out <- matrix(0, R, p) # Initialize an empty matrix to store the values
  beta <- rep(0, p) # Initial values
  logp <- logpost(beta, y, X)
  epsilon <- 1e-6 # Inital value for the covariance matrix
  # Initial matrix S
  S <- diag(epsilon, p)
  Sigma_r <- diag(0, p)
  mu_r <- beta
  for (r in 1:(burn_in + R)) {
    # Updating the covariance matrix
    if(r > 1){
      Sigma_r <- (r - 2) / (r - 1) * Sigma_r + tcrossprod(beta - mu_r) / r
      mu_r <- (r - 1) / r * mu_r + beta / r
      S <- 2.38^2 * Sigma_r / p + diag(epsilon, p)
    }
    # Eigen-decomposition
    eig <- eigen(S, symmetric = TRUE)
    A1 <- t(eig$vectors) * sqrt(eig$values)
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
set.seed(123)
R=1000000
burn_in=10000
# Running the MCMC
time_in_sec <- system.time(fit_MCMC_2<-as.mcmc(RMH_Adaptive(R = R, burn_in = burn_in, y, X)))
# > time_in_sec
# 用户  系统  流逝 
# 23.22  0.22 53.79

# Diagnostic
summary(effectiveSize(fit_MCMC_2 )) # Effective sample size
summary(R / effectiveSize(fit_MCMC_2 )) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_2)) # Acceptance rate
autocorrelation <- acf(fit_MCMC_2, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
# > summary(effectiveSize(fit_MCMC_2 )) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 37690   38473   38790   38995   39143   41112 
# > summary(R / effectiveSize(fit_MCMC_2 )) # Integrated autocorrelation time
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 24.32   25.55   25.78   25.66   25.99   26.53 
# > summary(1 - rejectionRate(fit_MCMC_2)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.2552  0.2552  0.2552  0.2552  0.2552  0.2552 
# > autocorrelation <- acf(fit_MCMC_2, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation
# [1] 0.9234157
#trace_plot and acf_plot
beta1_chain <- fit_MCMC_2[1:10000, 1]
# Plot Trace and ACF
par(mfrow=c(1,2))  # Set up the plotting area to have 1 row and 2 columns
traceplot(as.mcmc(beta1_chain), main="", xlab = "Iteration", ylab = "", ylim=c(-1.6,-0.5))
acf(beta1_chain, xlim=c(0,60),ylim=c(0,1),main = "")
mtext("AM", side=3, line=-2, outer=TRUE, cex=1.5)
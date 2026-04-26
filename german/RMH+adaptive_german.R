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
# 用户   系统   流逝 
# 42.14   0.28 121.02
# Diagnostic
summary(effectiveSize(fit_MCMC_2 )) # Effective sample size
summary(R / effectiveSize(fit_MCMC_2 )) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_2)) # Acceptance rate
autocorrelation <- acf(fit_MCMC_2, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
# > # Diagnostic
#   > summary(effectiveSize(fit_MCMC_2 )) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 11205   11655   11933   12256   12440   16643 
# > summary(R / effectiveSize(fit_MCMC_2 )) # Integrated autocorrelation time
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 60.08   80.39   83.80   82.24   85.80   89.24 
# > summary(1 - rejectionRate(fit_MCMC_2)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.2245  0.2245  0.2245  0.2245  0.2245  0.2245 
# > autocorrelation <- acf(fit_MCMC_2, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation
# [1] 0.9592332
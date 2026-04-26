library(MCMCpack)
Pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
y <- as.numeric(Pima$type == "Yes") # Binary outcome
X <- model.matrix(type ~ . - 1, data = Pima) # Design matrix
X <- cbind(1, scale(X)) # Standardize the design matrix, add the intercept
# Logit model
fit_logit <- glm(y ~ X - 1, family = binomial(link = "logit"), data = Pima)
#CGVB for Posterior
loglik <- function(thetax, y, X) {
  eta <- c(X %*% thetax)
  sum(y * eta - log(1 + exp(eta)))
}
# Logposterior
logpost <- function(beta, y, X) {
  loglik(beta, y, X) + sum(dnorm(beta, 0, 10, log = T))
}
# Logposterior
h_theta <- function(thetax, y, X) {
  loglik(thetax, y, X) + sum(dnorm(thetax, 0, 10, log = T))
}
#h_lam_theta
h<-function(thetax,y,X,L,mu){
  invL = solve(L)
  return(h_theta(thetax,y,X)+log(det(L)) + 0.5 * t(thetax - mu)%*%t(invL)%*%invL%*%(thetax - mu))
}
# Gradient of the logposterior
lgradient <- function(thetax, y, X) {
  probs <- plogis(c(X %*% thetax))
  loglik_gr <- c(crossprod(X, y - probs))
  prior_gr <- -thetax / 100
  loglik_gr + prior_gr
}
Gthetah = function(thetax,y,X, L, mu) #gradient with respect to theta of h function
{
  invL = solve(L)
  return(lgradient(thetax,y,X) + t(invL)%*%invL%*%(thetax - mu))
}
vechab = function(a, b)  #function to calculate the vectorization of the product of two matrices
{
  M = a%*%t(b)
  n = nrow(M)
  vecl = rep(0, n * (n + 1) / 2)
  count = 0
  for(nr in 1:n)
  {
    vecl[(count + 1):(count + n - nr + 1)] = M[nr:n, nr]
    count = count + n - nr + 1
  }
  return(vecl)
}
#initialization
patience = 0 
maxP = 50
beta1 = 0.9
beta2 = 0.9
e0 = 0.01
tau = 1000
p = 8
q = p * (p + 1) / 2
glen = p + q
TI = 5000
Lambda = matrix(0, TI, glen)
LB = rep(0, TI)
LBW = rep(0, TI)
k = p
start.time <- Sys.time()
for(i in 1:p)
{
  Lambda[1, (k + 1)] = 1
  k = k + p - i + 1
}
S = 100
epsilon = matrix(0, S, p)
theta = matrix(0, S, p)
tw = 50
t = 1
#iteration 
system.time({
  while(t < TI && patience < maxP)
  {
    #calculate mu and L from lambda
    mu = rep(0, p)
    L = matrix(0, p, p)
    mu = Lambda[t, 1:p]
    k = p
    for(i in 1:p)
    {
      L[i:p, i] = Lambda[t, (k + 1):(k + p - i + 1)]
      k = k + p - i + 1
    }
    #sample epsilon and calculate theta
    for(i in 1:S)
    {
      for(j in 1:p)
        epsilon[i, j] = rnorm(1)
      theta[i, ] = mu + L%*%epsilon[i, ]
    }
    #calculate LB and gradient of LB with respect to lambda
    gLB = rep(0, glen)
    for(i in 1:S)
    {
      LB[t] = LB[t] + h(theta[i, ],y,X,L,mu) / S
      gs = Gthetah(theta[i, ],y,X, L, mu)
      gLB[1:p] = gLB[1:p] + gs / S
      gLB[(p + 1):glen] = gLB[(p + 1):glen] + vechab(gs, epsilon[i, ]) / S
    }
    vLB = gLB^2
    if(t == 1) { 
      gbar = gLB
      vbar = vLB
    }
    #calculate adaptive gradient 
    else {
      gbar = beta1 * gbar + (1 - beta1) * gLB
      vbar = beta2 * vbar + (1 - beta2) * vLB
    }
    #calculate moving averaged LB
    if(t >= tw) {
      for(l in 1:tw)
      {
        LBW[t] = LBW[t] + LB[t - l + 1] / tw
      }
      if (LBW[t] >= max(LBW[tw:(t - 1)])) patience = 0
      else patience = patience + 1
    }
    #update lambda
    alpha = min(e0, e0 * tau / t)
    Lambda[t + 1, ] = Lambda[t, ] + alpha * (gbar / sqrt(vbar))
    t = t + 1
  }
})
#choose lambda corresponding to the largest moving averaged LB
index = which.max(LBW[tw:(t - 1)])
#calculate mu, L and Sigma from the optimal lambda
mu = rep(0, p)
L = matrix(0, p, p)
mu = Lambda[index, 1:p]
k = p
for(i in 1:p)
{
  L[i:p, i] = Lambda[index, (k + 1):(k + p - i + 1)]
  k = k + p - i + 1
}
Sigma = L%*%t(L)
mu
Sigma
# 用户 系统 流逝 
# 0.97 0.03 2.63 

#eig <- eigen(Sigma, symmetric = TRUE)
#eigenvalues <- eig$values
#L<-ceiling(sqrt(max(eigenvalues))/min(eigenvalues))
#epsilon<-sqrt(min(eigenvalues))
HMC <- function(R, burn_in, y, X, epsilon, S, L ) {
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
    if (r > burn_in) {
      out[r - burn_in, ] <- beta
    }
  }
  out
}
set.seed(123)
# Covariance matrix is selected via laplace approximation
S=Sigma
R=1000000
burn_in=10000
epsilon = 0.16
L=10
system.time(fit_MCMC_4 <- as.mcmc(HMC(R = R, burn_in = burn_in, y, X, epsilon, S, L)))
samples=fit_MCMC_4
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
# Diagnostic
summary(effectiveSize(fit_MCMC_4)) # Effective sample size
summary(R / effectiveSize(fit_MCMC_4)) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC_4)) # Acceptance rate
samples=fit_MCMC_4
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
# > # Diagnostic
#   > summary(effectiveSize(fit_MCMC_4)) # Effective sample size
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 920654  950170  972307  990571 1009266 1097055 
# > summary(R / effectiveSize(fit_MCMC_4)) # Integrated autocorrelation time
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.9115  0.9922  1.0286  1.0129  1.0524  1.0862 
# > summary(1 - rejectionRate(fit_MCMC_4)) # Acceptance rate
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.9932  0.9932  0.9932  0.9932  0.9932  0.9932 
# > samples=fit_MCMC_4
# > autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
# > autocorrelation 
# [1] 0.01713187
#trace_plot and acf_plot
beta1_chain <- fit_MCMC_4[1:10000, 1]
# Plot Trace and ACF
par(mfrow=c(1,2))  # Set up the plotting area to have 1 row and 2 columns
traceplot(as.mcmc(beta1_chain), main="", xlab = "Iteration", ylab = "", ylim=c(-1.6,-0.5))
acf(beta1_chain, xlim=c(0,60),ylim=c(0,1),main = "")
mtext("HMC-GVI", side=3, line=-2, outer=TRUE, cex=1.5)
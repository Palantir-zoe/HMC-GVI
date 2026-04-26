set.seed(66)
p=100
mu=rep(0,p)
#cov=matrix(c(1,1.8,1.8,4),nrow=2,ncol=2,byrow=TRUE)
#Smatrix =riwish(p, diag(rep(1,p)))
mat = matrix(0.8,nrow=p,ncol=p)
diag(mat) = rgamma(p,shape=2,scale=3)
cov=mat

log_tar = function(theta, cov, mu) 
{
  inv_cov = solve(cov)
  return(-0.5 *t(theta - mu)%*%inv_cov%*%(theta - mu))
}
#gradient
G_tar = function(theta, cov, mu) 
{
  inv_cov = solve(cov)
  return(c(-inv_cov %*% (theta - mu))) # transpose the first matrix)
}
#HMC burn_in
HMC<-function(burn_in,theta,cov,mu,S,epsilon,L){
  p <- ncol(cov)
  out <- matrix(0,burn_in,p)#initialize an empty matrix to store samples
  theta <- rep(0,p)#initial values
  logp<-log_tar(theta,cov,mu)#initial log probability
  S1<-solve(S)
  A1<-chol(S1)
  
  
  #starting the Gibbs sampling
  for ( r in 1:burn_in){
    P <- c(crossprod(A1,rnorm(p))) #auxiliary variables
    logK <- c(P %*% S %*% P/2) #kinetic energy at the beginning of the trajectory
    
    #Make a half step for momentum at the beginning
    theta_new <- theta
    Pnew <- P + epsilon * G_tar(theta_new,cov,mu)/2
    
    #Alternate full steps for position and momentum
    for (l in 1:L){
      #Make a full step for the position
      theta_new <- theta_new + epsilon * c(S %*% Pnew)
      #make a full step for the momentum, except at the end of trajectory
      if (l != L) Pnew <- Pnew + epsilon * G_tar(theta_new,cov,mu)
    }
    #Make a half step for momentum at the end
    Pnew <- Pnew +epsilon * G_tar(theta_new,cov,mu)/2 
    
    #Negate momentum at the end of trajectory to make the proposal symmetric
    Pnew <- -Pnew
    
    #Evaluate potential and kinetic energies at the end of trajectory
    logpnew <- log_tar(theta_new,cov,mu)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    
    #Accept or reject the state at end of trajectory
    #the position at the end of the trajectory or the initial position
    if (runif(1) < exp(logpnew - logp + logK - logKnew)){
      logp <- logpnew
      theta <- theta_new #accept the value
    }
    #store the values after the burn_in period
    out[r,]<- theta
  }
  out
}
set.seed(123)
epsilon <- 0.16# After some trial ad error
L <- 10
# Covariance matrix is selected via laplace approximation
R <- 1000000 # Number of retained samples
burn_in <- 10000 # Burn-in period
S <- diag(p)
# Running the MCMC
system.time(fit_MCMC_1 <- as.mcmc(HMC(burn_in=burn_in,theta,cov=cov,mu=mu,S=S,epsilon=epsilon,L=L)))
samples=fit_MCMC_1
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation
colMeans(fit_MCMC_1)
Sigma1=cov(fit_MCMC_1)

#HMC
HMC<-function(R,burn_in,theta,cov,mu,S,epsilon,L){
  p <- ncol(cov)
  out <- matrix(0,R,p)#initialize an empty matrix to store samples
  theta <- rep(0,p)#initial values
  logp<-log_tar(theta,cov,mu)#initial log probability
  S1<-solve(S)
  A1<-chol(S1)
  
  
  #starting the Gibbs sampling
  for ( r in 1:(burn_in+R)){
    P <- c(crossprod(A1,rnorm(p))) #auxiliary variables
    logK <- c(P %*% S %*% P/2) #kinetic energy at the beginning of the trajectory
    
    #Make a half step for momentum at the beginning
    theta_new <- theta
    Pnew <- P + epsilon * G_tar(theta_new,cov,mu)/2
    
    #Alternate full steps for position and momentum
    for (l in 1:L){
      #Make a full step for the position
      theta_new <- theta_new + epsilon * c(S %*% Pnew)
      #make a full step for the momentum, except at the end of trajectory
      if (l != L) Pnew <- Pnew + epsilon * G_tar(theta_new,cov,mu)
    }
    #Make a half step for momentum at the end
    Pnew <- Pnew +epsilon * G_tar(theta_new,cov,mu)/2 
   
     #Negate momentum at the end of trajectory to make the proposal symmetric
    Pnew <- -Pnew
    
    #Evaluate potential and kinetic energies at the end of trajectory
    logpnew <- log_tar(theta_new,cov,mu)
    logKnew <- Pnew %*% S %*% Pnew / 2 
    
    #Accept or reject the state at end of trajectory
    #the position at the end of the trajectory or the initial position
    if (runif(1) < exp(logpnew - logp + logK - logKnew)){
      logp <- logpnew
      theta <- theta_new #accept the value
    }
    #store the values after the burn_in period
    if (r>burn_in){
      out[r-burn_in,]<- theta
    }
  }
  out
}
set.seed(123)
epsilon <- 0.16# After some trial ad error
L <- 10
# Covariance matrix is selected via laplace approximation
R <- 1000000 # Number of retained samples
burn_in <- 10000 # Burn-in period
S <- Sigma1
# Running the MCMC
system.time(fit_MCMC <- as.mcmc(HMC(R=R,burn_in=burn_in,theta,cov=cov,mu=mu,S= Sigma1,epsilon=epsilon,L=L)))
samples=fit_MCMC
autocorrelation <- acf(samples, lag.max = 1, plot = FALSE)$acf[2]
autocorrelation



colMeans(fit_MCMC)
cov(fit_MCMC)
#time_in_sec <- system.time(fit_MCMC_1 <- as.mcmc(HMC(R,burn_in,theta,S,mu,Sigma,epsilon,L)))
# Diagnostic
summary(effectiveSize(fit_MCMC )) # Effective sample size
summary(R / effectiveSize(fit_MCMC )) # Integrated autocorrelation time
summary(1 - rejectionRate(fit_MCMC )) # Acceptance rate

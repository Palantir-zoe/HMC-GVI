#Target: multivariate gaussian distribution
#Proposal: Normal
#Metropolis within Gibbs
library(MCMCpack)
p = 50
m = rep(0, p) #true mean
##Smatrix =diag(rep(1),p,p)#true variance
Smatrix = riwish(p, diag(rep(1,p))) 
#Smatrix = diag(1, p, p) 
diagS = diag(Smatrix)
normal_pdf <- function(thetax) {
  -0.5 * sum((thetax - m)^2 / diagS)
}
RMH_Gibbs <- function(R,se){
  
  out <- matrix(0,R,p)
  thetax <- thetax_new <-rep(0.5,p)#initial values
  logp<-normal_pdf(thetax)
  
  for(r in 1:R){
    for (j in 1:p){
      thetax_new[j]<-thetax[j]+rnorm(1,0,se)
      logp_new <- normal_pdf( thetax_new )
      alpha <- min(1,exp(logp_new-logp))
      if (runif(1)<alpha){
        logp<-logp_new
        thetax[j]<-thetax_new[j]#accept the value
      }
    }
    #store the values after the burn_in period
    out[r,]<-thetax
  }
  out
}
se = 0.77
R = 25000
system.time(fit_mcmc <- as.mcmc(RMH_Gibbs(R,se)))
mu = colMeans(fit_mcmc )
Sigma = cov(fit_mcmc )
###data###
#install.packages("aplore3")
library(aplore3)
data("polypharm")
polypharm
write.csv(polypharm, "polypharm.csv")
#
n = 500
ni = 7
k = 8
p = 1
#
Poly_id = c(polypharm[,1])
Poly_xx = matrix(data = 1, nrow = n * ni, ncol = k)
Poly_y = numeric(n * ni)
#
Poly_xx[,2] = as.numeric(c(polypharm$gender == 'Male'))
Poly_xx[,3] = as.numeric(c(polypharm$race != 'White'))
Poly_xx[,4] = as.numeric(c(polypharm$age))
Poly_xx[,4] = log(Poly_xx[,4]/10)
Poly_xx[,5] = as.numeric(c(polypharm$mhv4 == '1-5'))
Poly_xx[,6] = as.numeric(c(polypharm$mhv4 == '6-14'))
Poly_xx[,7] = as.numeric(c(polypharm$mhv4 == '> 14'))
Poly_xx[,8] = as.numeric(c(polypharm$inptmhv3 != '0'))
Poly_y = as.numeric(c(polypharm$polypharmacy == 'Yes'))


###log posterior & gradient of log posterior###

sb = 10
sz = 10

logh = function(thetax)
{
  # log posterior density
  uu = thetax[1:n] # random effect
  bt = thetax[(n + 1):(n + k)] # fixed effect,coefficient
  zt = thetax[n + k + 1] #log(sd) for random effect
  uu1 = rep(uu, each = ni)
  e1 = c(Poly_xx %*% bt) + uu1
  y = sum(Poly_y * e1 - log(1 + exp(e1))) - 0.5 * exp(-2 * zt) * sum(uu^2) 
    - sum(bt^2) / (2 * sb^2) - zt^2 / (2 * sz^2) - n * zt
  
  return(y)
}

e3 = colSums(Poly_y * Poly_xx)
grad_logh = function(thetax)
{
  # the gradient of log posterior density
  uu = thetax[1:n] 
  bt = thetax[(n + 1):(n + k)] 
  zt = thetax[n + k + 1]
  uu1 = rep(uu, each = ni)
  e2 = exp(c(Poly_xx %*% bt) + uu1) 
  duu = colSums(matrix(Poly_y - e2 / (1 + e2), nrow = ni)) - exp(-2 * zt) * uu
  dbt = e3 - 
    colSums(e2 / (1 + e2) * Poly_xx) - 
    bt / (sb^2)
  dzt = exp(-2 * zt) * sum(uu^2) - zt / (sz^2) - n
  return(c(duu, dbt, dzt))
}

###variational algorithm###
rho = 0.95
eps = 10^(-6)
theta_length = n + k + p
mu = rep(0, theta_length); 
T_mat = diag(1, theta_length, theta_length)
T_prime_mat = T_mat; diag(T_prime_mat) = log(diag(T_prime_mat));
Eg2mu = Edelta2mu = rep(0, theta_length)
Eg2T_prime = Edelta2T_prime = matrix(0, theta_length, theta_length)

TI = 20000
t = 1
LB = rep(0, TI)
LBW = Lmax = 0  
Ft = 10000
M = 50
tm = 0

#iteration 
system.time(while((t < TI) && (tm <= M))
{
  s = rnorm(theta_length)
  s1 = s[1:n]; s2 = s[(n + 1):theta_length];
  A = T_mat[1:n, 1:n]; C = T_mat[(n + 1):theta_length, 1:n]; D = T_mat[(n + 1):theta_length, (n + 1):theta_length];
  vec_A = diag(A); vec_invA = 1 / vec_A; invA = diag(vec_invA, n, n);
  invD = solve(D); invDCinvA = (invD%*%C) * vec_invA;
  tinvTs = c(vec_invA * s1 - as.vector(t(invDCinvA)%*%s2), as.vector(t(invD)%*%s2))
  theta = mu + tinvTs
  LB[t] = logh(theta) + theta_length / 2 * log(2 *pi) 
        - sum(log(abs(vec_A))) - log(abs(det(D)))
        + 0.5 * as.numeric(t(s)%*%s)
  if(t >= Ft){
    LBW = mean(LB[(t - (Ft - 1)) : t])
    if(t == Ft) Lmax = LBW
    else{
      if(LBW < Lmax) tm = tm + 1
      else{
        tm = 0
        Lmax = LBW
      }
    }
  }
  Ts = c(vec_A * s1, as.vector(C%*%s1 + D%*%s2))
  gmu = grad_logh(theta) + Ts
  Eg2mu = rho * Eg2mu + (1 - rho) * gmu^2
  deltamu = sqrt(Edelta2mu + eps) / sqrt(Eg2mu + eps) * gmu
  Edelta2mu = rho * Edelta2mu + (1 - rho) * deltamu^2
  mu = mu + deltamu
  gmu1 = gmu[1:n]; gmu2 = gmu[(n + 1):theta_length];
  invTgmu = c(vec_invA * gmu1, as.vector(-invDCinvA%*%gmu1 + invD%*%gmu2))
  gT_prime = -tinvTs%*%t(invTgmu)
  diag(gT_prime) = diag(gT_prime) * diag(T_mat)
  Eg2T_prime = rho * Eg2T_prime + (1 - rho) * gT_prime^2
  deltaT_prime = sqrt(Edelta2T_prime + eps) / sqrt(Eg2T_prime + eps) * gT_prime
  Edelta2T_prime = rho * Edelta2T_prime + (1 - rho) * deltaT_prime^2
  T_prime_mat = T_prime_mat + deltaT_prime
  T_prime_mat[1:n, 1:n] = diag(diag(T_prime_mat[1:n, 1:n]), n, n)
  T_prime_mat[upper.tri(T_prime_mat, diag = FALSE)] = 0
  T_mat = T_prime_mat; diag(T_mat) = exp(diag(T_mat));
  t = t + 1
})
# 用户   系统   流逝 
# 53.33  26.58 154.34

###result###
#mu#
mu

#Sigma#
A = T_mat[1:n, 1:n]; C = T_mat[(n + 1):theta_length, 1:n]; D = T_mat[(n + 1):theta_length, (n + 1):theta_length];
vec_A = diag(A); vec_invA = 1 / vec_A; invA = diag(vec_invA, n, n);
invD = solve(D); invDCinvA = (invD%*%C) * vec_invA;

Sigma = matrix(0, theta_length, theta_length)
Sigma[1:n, 1:n] = diag(vec_invA^2, n, n) + t(invDCinvA)%*%invDCinvA
Sigma[(n + 1):theta_length, 1:n] = -t(invD)%*%invDCinvA
Sigma[1:n, (n + 1):theta_length] = t(Sigma[(n + 1):theta_length, 1:n])
Sigma[(n + 1):theta_length, (n + 1):theta_length] = t(invD)%*%invD
Sigma
# #SD of beta and ksi#
# sqrt(diag(t(invD)%*%invD)) 
# 
# -4.253088427
# 0.743127835
# -0.656772809
# 2.617216709
# 0.327792574
# 1.201056683
# 1.740464809
# 0.897933993
# 0.919866019
# 
# sqrt(c(
#   0.165946475,
#   0.118927963,
#   0.148880819,
#   0.093467269,
#   0.083812642,
#   0.087438961,
#   0.09098393,
#   0.065434547,
#   0.004500405
# ))
mu1=mu
Sigma1=Sigma

mu2=mu
Sigma2=Sigma

mu3=mu
Sigma3=Sigma

mu_gvi=(mu1+mu2+mu3)/3
Sigma_gvi =(Sigma1+Sigma2+Sigma3)/3

write.csv(mu_gvi, "mu_gvi.csv", row.names = FALSE)
write.csv(Sigma_gvi, "Sigma_gvi.csv", row.names = FALSE)

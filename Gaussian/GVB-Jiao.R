####GVB####
m = c(1,2) #true mean
S = matrix(c(1, 1.8, 1.8, 4), 2, 2) #true variance
invS = solve(S)
detS = det(S)
h = function(thetax, L, mu) #h function
{
  invL = solve(L)
  return(-0.5 * log(detS) - 0.5 * t(thetax - m)%*%invS%*%(thetax - m) + log(det(L)) + 0.5 * t(thetax - mu)%*%t(invL)%*%invL%*%(thetax - mu))
}
Gthetah = function(thetax, L, mu) #gradient with respect to theta of h function
{
  invL = solve(L)
  return(-invS%*%(thetax - m) + t(invL)%*%invL%*%(thetax - mu))
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
p = 2
q = p * (p + 1) / 2
glen = p + q
TI = 5000
Lambda = matrix(0, TI, glen)
LB = rep(0, TI)
LBW = rep(0, TI)
k = p
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
    {
      epsilon[i, j] = rnorm(1)
    }
    theta[i, ] = mu + L%*%epsilon[i, ]
  }
  #calculate LB and gradient of LB with respect to lambda
  gLB = rep(0, glen)
  for(i in 1:S)
  {
    LB[t] = LB[t] + h(theta[i, ], L, mu) / S
    gs = Gthetah(theta[i, ], L, mu)
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

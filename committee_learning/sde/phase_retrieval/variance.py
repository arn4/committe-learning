
def _variance_q(q,m,rho,gamma,noise):
  return (
    320*m**4 - 2688*m**2*q**2 + 1536*q**4 - 33024*m**4*q*gamma + 77376*m**2*q**3*gamma - 28800*q**5*gamma -
    46080*m**6*gamma**2 + 474624*m**4*q**2*gamma**2 - 593280*m**2*q**4*gamma**2 + 162720*q**6*gamma**2 + 48*q**2*noise +
    1088*m**2*q*gamma*noise - 1344*q**3*gamma*noise + 2304*m**4*gamma**2*noise - 16512*m**2*q**2*gamma**2*noise + 9600*q**4*gamma**2*noise +
    128*q**2*gamma**2*noise**2 + 1088*m**2*q*rho - 384*q**3*rho + 10752*m**4*gamma*rho - 49536*m**2*q**2*gamma*rho +
    9024*q**4*gamma*rho - 336384*m**4*q*gamma**2*rho + 473472*m**2*q**3*gamma**2*rho - 57600*q**5*gamma**2*rho +
    256*q**2*gamma*noise*rho + 6528*m**2*q*gamma**2*noise*rho - 2688*q**3*gamma**2*noise*rho + 128*q**2*rho**2 + 16704*m**2*q*gamma*rho**2 -
    3840*q**3*gamma*rho**2 + 78336*m**4*gamma**2*rho**2 - 254592*m**2*q**2*gamma**2*rho**2 + 28224*q**4*gamma**2*rho**2 +
    768*q**2*gamma**2*noise*rho**2 + 1344*q**2*gamma*rho**3 + 79488*m**2*q*gamma**2*rho**3 - 13824*q**3*gamma**2*rho**3 + 
    4896*q**2*gamma**2*rho**4
  )

def _variance_m(q,m,rho,gamma,noise):
  return (
    -192*m**4*gamma**2 + 324*m**2*q**2*gamma**2 + 8*m**2*gamma**2*noise - 504*m**2*q*gamma**2*rho + 60*q**3*gamma**2*rho + 
    4*q*gamma**2*noise*rho + 324*m**2*gamma**2*rho**2 - 72*q**2*gamma**2*rho**2 + 60*q*gamma**2*rho**3
  )

def _covariance_qm(q,m,rho,gamma,noise):
  return (
    -912*m**3*q*gamma + 768*m*q**3*gamma - 2880*m**5*gamma**2 + 14544*m**3*q**2*gamma**2 - 7200*m*q**4*gamma**2 + 
    24*m*q*gamma*noise + 144*m**3*gamma**2*noise - 336*m*q**2*gamma**2*noise + 432*m**3*gamma*rho - 624*m*q**2*gamma*rho - 
    13536*m**3*q*gamma**2*rho + 7056*m*q**3*gamma**2*rho + 192*m*q*gamma**2*noise*rho + 336*m*q*gamma*rho**2 + 
    4752*m**3*gamma**2*rho**2 - 5184*m*q**2*gamma**2*rho**2 + 2448*m*q*gamma**2*rho**3
  )




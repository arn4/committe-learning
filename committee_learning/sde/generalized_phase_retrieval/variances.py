import numpy as np

from ..._cython.numpy_extra import symmetrize

def spherical_p1variance(q, m, gamma, noise):
    return symmetrize(np.array([
        # Var[q]
        1280 - 11136*gamma + 31104*gamma**2 + 48*noise - 544*gamma*noise + 1920*gamma**2*noise + 32*gamma**2*noise**2 - 1600*m**2 + 22272*gamma*m**2 - 73728*gamma**2*m**2 + 544*gamma*noise*m**2 - 2496*gamma**2*noise*m**2 + 320*m**4 - 11136*gamma*m**4 + 54144*gamma**2*m**4 + 576*gamma**2*noise*m**4 - 11520*gamma**2*m**6,
        # Cov[q, m]
        480*m - 1440*gamma*m + 24*noise*m - 72*gamma*noise*m - 480*m**3 + 2880*gamma*m**3 + 72*gamma*noise*m**3 - 1440*gamma*m**5,
        # Var[m]
        48 + 4*noise + 144*m**2 + 8*noise*m**2 - 192*m**4
    ]))

def spherical_p2variance(q11, q12, q22, m1, m2, gamma, noise):
    return symmetrize(np.array([
        # Var[q11]
        416 - 1648*gamma + 3032*gamma**2 + 48*noise - 272*gamma*noise + 608*gamma**2*noise + 32*gamma**2*noise**2 - 416*m1**2 + 6432*gamma*m1**2 - 4384*gamma**2*m1**2 + 544*gamma*noise*m1**2 - 704*gamma**2*noise*m1**2 + 320*m1**4 - 3392*gamma*m1**4 + 9056*gamma**2*m1**4 + 576*gamma**2*noise*m1**4 - 5760*gamma**2*m1**6 - 96*m2**2 + 
        256*gamma*m2**2 - 1504*gamma**2*m2**2 - 128*gamma**2*noise*m2**2 - 3392*gamma*m1**2*m2**2 + 832*gamma**2*m1**2*m2**2 - 5760*gamma**2*m1**4*m2**2 + 416*gamma**2*m2**4 - 768*m1*m2*q12 + 5056*gamma*m1*m2*q12 - 18112*gamma**2*m1*m2*q12 - 1088*gamma**2*noise*m1*m2*q12 - 
        4352*gamma*m1**3*m2*q12 + 18304*gamma**2*m1**3*m2*q12 + 6784*gamma**2*m1*m2**3*q12 + 784*q12**2 - 7008*gamma*q12**2 + 15776*gamma**2*q12**2 - 272*gamma*noise*q12**2 + 1168*gamma**2*noise*q12**2 - 320*m1**2*q12**2 + 6656*gamma*m1**2*q12**2 - 22784*gamma**2*m1**2*q12**2 - 
        576*gamma**2*noise*m1**2*q12**2 + 8640*gamma**2*m1**4*q12**2 + 1696*gamma*m2**2*q12**2 - 5504*gamma**2*m2**2*q12**2 + 10112*gamma**2*m1**2*m2**2*q12**2 + 2176*gamma*m1*m2*q12**3 - 15680*gamma**2*m1*m2*q12**3 + 80*q12**4 - 2480*gamma*q12**4 + 11576*gamma**2*q12**4 + 
        144*gamma**2*noise*q12**4 - 4320*gamma**2*m1**2*q12**4 - 1440*gamma**2*m2**2*q12**4 + 720*gamma**2*q12**6,
        # Cov[q11, q12]
        -160*m1*m2 + 3088*gamma*m1*m2 - 1440*gamma**2*m1*m2 + 272*gamma*noise*m1*m2 - 288*gamma**2*noise*m1*m2 + 320*m1**3*m2 - 1696*gamma*m1**3*m2 + 8640*gamma**2*m1**3*m2 + 576*gamma**2*noise*m1**3*m2 - 5760*gamma**2*m1**5*m2 - 1696*gamma*m1*m2**3 - 
        5760*gamma**2*m1**3*m2**3 + 688*q12 - 3736*gamma*q12 + 6840*gamma**2*q12 + 48*noise*q12 - 408*gamma*noise*q12 + 1024*gamma**2*noise*q12 + 32*gamma**2*noise**2*q12 - 608*m1**2*q12 + 5824*gamma*m1**2*q12 - 12048*gamma**2*m1**2*q12 + 272*gamma*noise*m1**2*q12 - 976*gamma**2*noise*m1**2*q12 - 
        2784*gamma*m1**4*q12 + 4992*gamma**2*m1**4*q12 - 288*m2**2*q12 + 1648*gamma*m2**2*q12 - 6288*gamma**2*m2**2*q12 - 400*gamma**2*noise*m2**2*q12 - 4960*gamma*m1**2*m2**2*q12 + 12864*gamma**2*m1**2*m2**2*q12 + 2112*gamma**2*m2**4*q12 - 544*m1*m2*q12**2 + 
        7536*gamma*m1*m2*q12**2 - 25632*gamma**2*m1*m2*q12**2 - 832*gamma**2*noise*m1*m2*q12**2 + 17088*gamma**2*m1**3*m2*q12**2 + 8448*gamma**2*m1*m2**3*q12**2 + 592*q12**3 - 6704*gamma*q12**3 + 18768*gamma**2*q12**3 - 136*gamma*noise*q12**3 + 896*gamma**2*noise*q12**3 + 
        2784*gamma*m1**2*q12**3 - 13488*gamma**2*m1**2*q12**3 + 1392*gamma*m2**2*q12**3 - 7728*gamma**2*m2**2*q12**3 - 7104*gamma**2*m1*m2*q12**4 - 696*gamma*q12**5 + 5496*gamma**2*q12**5,
        # Cov[q11, q22]
        128 - 496*gamma + 944*gamma**2 + 16*noise - 80*gamma*noise + 176*gamma**2*noise + 8*gamma**2*noise**2 - 64*m1**2 + 1040*gamma*m1**2 - 928*gamma**2*m1**2 + 80*gamma*noise*m1**2 - 128*gamma**2*noise*m1**2 - 544*gamma*m1**4 + 128*gamma**2*m1**4 - 64*m2**2 + 1040*gamma*m2**2 - 928*gamma**2*m2**2 + 80*gamma*noise*m2**2 - 128*gamma**2*noise*m2**2 + 
        320*m1**2*m2**2 - 1088*gamma*m1**2*m2**2 + 8896*gamma**2*m1**2*m2**2 + 576*gamma**2*noise*m1**2*m2**2 - 5760*gamma**2*m1**4*m2**2 - 544*gamma*m2**4 + 128*gamma**2*m2**4 - 5760*gamma**2*m1**2*m2**4 - 768*m1*m2*q12 + 7360*gamma*m1*m2*q12 - 
        13504*gamma**2*m1*m2*q12 + 384*gamma*noise*m1*m2*q12 - 1088*gamma**2*noise*m1*m2*q12 - 4480*gamma*m1**3*m2*q12 + 7936*gamma**2*m1**3*m2*q12 - 4480*gamma*m1*m2**3*q12 + 7936*gamma**2*m1*m2**3*q12 + 976*q12**2 - 7008*gamma*q12**2 + 14552*gamma**2*q12**2 + 
        32*noise*q12**2 - 464*gamma*noise*q12**2 + 1456*gamma**2*noise*q12**2 + 24*gamma**2*noise**2*q12**2 - 352*m1**2*q12**2 + 4176*gamma*m1**2*q12**2 - 14720*gamma**2*m1**2*q12**2 - 576*gamma**2*noise*m1**2*q12**2 + 4320*gamma**2*m1**4*q12**2 - 352*m2**2*q12**2 + 4176*gamma*m2**2*q12**2 - 
        14720*gamma**2*m2**2*q12**2 - 576*gamma**2*noise*m2**2*q12**2 + 20480*gamma**2*m1**2*m2**2*q12**2 + 4320*gamma**2*m2**4*q12**2 + 4480*gamma*m1*m2*q12**3 - 20288*gamma**2*m1*m2*q12**3 + 176*q12**4 - 3632*gamma*q12**4 + 14528*gamma**2*q12**4 + 288*gamma**2*noise*q12**4 - 
        4320*gamma**2*m1**2*q12**4 - 4320*gamma**2*m2**2*q12**4 + 1080*gamma**2*q12**6,
        # Cov[q11, m1]
        224*m1 + 196*gamma*m1 + 24*noise*m1 - 4*gamma*noise*m1 - 64*m1**3 + 872*gamma*m1**3 + 72*gamma*noise*m1**3 - 720*gamma*m1**5 - 144*m1*m2**2 - 208*gamma*m1*m2**2 - 720*gamma*m1**3*m2**2 + 40*m2*q12 - 376*gamma*m2*q12 - 32*gamma*noise*m2*q12 - 272*m1**2*m2*q12 + 
        672*gamma*m1**2*m2*q12 + 208*gamma*m2**3*q12 + 176*m1*q12**2 - 784*gamma*m1*q12**2 - 36*gamma*noise*m1*q12**2 + 720*gamma*m1**3*q12**2 + 616*gamma*m1*m2**2*q12**2 + 40*m2*q12**3 - 296*gamma*m2*q12**3 - 180*gamma*m1*q12**4,
        # Cov[q11, m2]
        80*m2 + 52*gamma*m2 + 8*noise*m2 - 4*gamma*noise*m2 + 32*m1**2*m2 + 1016*gamma*m1**2*m2 + 72*gamma*noise*m1**2*m2 - 720*gamma*m1**4*m2 - 48*m2**3 - 64*gamma*m2**3 - 720*gamma*m1**2*m2**3 + 184*m1*q12 - 232*gamma*m1*q12 + 16*noise*m1*q12 - 32*gamma*noise*m1*q12 - 
        176*m1**3*q12 + 64*gamma*m1**3*q12 - 288*m1*m2**2*q12 + 240*gamma*m1*m2**2*q12 + 128*m2*q12**2 - 640*gamma*m2*q12**2 - 36*gamma*noise*m2*q12**2 + 1264*gamma*m1**2*m2*q12**2 + 360*gamma*m2**3*q12**2 + 88*m1*q12**3 - 440*gamma*m1*q12**3 - 
        180*gamma*m2*q12**4,
        # Var[q12]
        144 - 576*gamma + 1044*gamma**2 + 16*noise - 96*gamma*noise + 216*gamma**2*noise + 12*gamma**2*noise**2 - 96*m1**2 + 1152*gamma*m1**2 - 1008*gamma**2*m1**2 + 96*gamma*noise*m1**2 - 144*gamma**2*noise*m1**2 - 576*gamma*m1**4 + 144*gamma**2*m1**4 - 96*m2**2 + 1152*gamma*m2**2 - 1008*gamma**2*m2**2 + 96*gamma*noise*m2**2 - 144*gamma**2*noise*m2**2 + 
        320*m1**2*m2**2 - 1152*gamma*m1**2*m2**2 + 8928*gamma**2*m1**2*m2**2 + 576*gamma**2*noise*m1**2*m2**2 - 5760*gamma**2*m1**4*m2**2 - 576*gamma*m2**4 + 144*gamma**2*m2**4 - 5760*gamma**2*m1**2*m2**4 - 640*m1*m2*q12 + 7200*gamma*m1*m2*q12 - 
        13824*gamma**2*m1*m2*q12 + 352*gamma*noise*m1*m2*q12 - 1152*gamma**2*noise*m1*m2*q12 - 4416*gamma*m1**3*m2*q12 + 8064*gamma**2*m1**3*m2*q12 - 4416*gamma*m1*m2**3*q12 + 8064*gamma**2*m1*m2**3*q12 + 944*q12**2 - 6912*gamma*q12**2 + 14508*gamma**2*q12**2 + 
        32*noise*q12**2 - 448*gamma*noise*q12**2 + 1440*gamma**2*noise*q12**2 + 20*gamma**2*noise**2*q12**2 - 384*m1**2*q12**2 + 4128*gamma*m1**2*q12**2 - 14400*gamma**2*m1**2*q12**2 - 528*gamma**2*noise*m1**2*q12**2 + 4176*gamma**2*m1**4*q12**2 - 384*m2**2*q12**2 + 4128*gamma*m2**2*q12**2 - 
        14400*gamma**2*m2**2*q12**2 - 528*gamma**2*noise*m2**2*q12**2 + 20448*gamma**2*m1**2*m2**2*q12**2 + 4176*gamma**2*m2**4*q12**2 + 4512*gamma*m1*m2*q12**3 - 20736*gamma**2*m1*m2*q12**3 + 192*q12**4 - 3648*gamma*q12**4 + 14508*gamma**2*q12**4 + 264*gamma**2*noise*q12**4 - 
        4176*gamma**2*m1**2*q12**4 - 4176*gamma**2*m2**2*q12**4 + 1044*gamma**2*q12**6,
        # Cov[q12, q22]
        -160*m1*m2 + 3088*gamma*m1*m2 - 1440*gamma**2*m1*m2 + 272*gamma*noise*m1*m2 - 288*gamma**2*noise*m1*m2 - 1696*gamma*m1**3*m2 + 320*m1*m2**3 - 1696*gamma*m1*m2**3 + 8640*gamma**2*m1*m2**3 + 576*gamma**2*noise*m1*m2**3 - 5760*gamma**2*m1**3*m2**3 - 
        5760*gamma**2*m1*m2**5 + 688*q12 - 3736*gamma*q12 + 6840*gamma**2*q12 + 48*noise*q12 - 408*gamma*noise*q12 + 1024*gamma**2*noise*q12 + 32*gamma**2*noise**2*q12 - 288*m1**2*q12 + 1648*gamma*m1**2*q12 - 6288*gamma**2*m1**2*q12 - 400*gamma**2*noise*m1**2*q12 + 2112*gamma**2*m1**4*q12 - 
        608*m2**2*q12 + 5824*gamma*m2**2*q12 - 12048*gamma**2*m2**2*q12 + 272*gamma*noise*m2**2*q12 - 976*gamma**2*noise*m2**2*q12 - 4960*gamma*m1**2*m2**2*q12 + 12864*gamma**2*m1**2*m2**2*q12 - 2784*gamma*m2**4*q12 + 4992*gamma**2*m2**4*q12 - 544*m1*m2*q12**2 + 
        7536*gamma*m1*m2*q12**2 - 25632*gamma**2*m1*m2*q12**2 - 832*gamma**2*noise*m1*m2*q12**2 + 8448*gamma**2*m1**3*m2*q12**2 + 17088*gamma**2*m1*m2**3*q12**2 + 592*q12**3 - 6704*gamma*q12**3 + 18768*gamma**2*q12**3 - 136*gamma*noise*q12**3 + 896*gamma**2*noise*q12**3 + 
        1392*gamma*m1**2*q12**3 - 7728*gamma**2*m1**2*q12**3 + 2784*gamma*m2**2*q12**3 - 13488*gamma**2*m2**2*q12**3 - 7104*gamma**2*m1*m2*q12**4 - 696*gamma*q12**5 + 5496*gamma**2*q12**5,
        # Cov[q12, m1]
        72*m2 + 72*gamma*m2 + 8*noise*m2 + 32*m1**2*m2 + 1008*gamma*m1**2*m2 + 72*gamma*noise*m1**2*m2 - 720*gamma*m1**4*m2 - 48*m2**3 - 72*gamma*m2**3 - 720*gamma*m1**2*m2**3 + 208*m1*q12 - 288*gamma*m1*q12 + 16*noise*m1*q12 - 40*gamma*noise*m1*q12 - 
        192*m1**3*q12 + 96*gamma*m1**3*q12 - 272*m1*m2**2*q12 + 240*gamma*m1*m2**2*q12 + 104*m2*q12**2 - 576*gamma*m2*q12**2 - 32*gamma*noise*m2*q12**2 + 1272*gamma*m1**2*m2*q12**2 + 336*gamma*m2**3*q12**2 + 96*m1*q12**3 - 480*gamma*m1*q12**3 - 
        168*gamma*m2*q12**4,
        # Cov[q12, m2]
        72*m1 + 72*gamma*m1 + 8*noise*m1 - 48*m1**3 - 72*gamma*m1**3 + 32*m1*m2**2 + 1008*gamma*m1*m2**2 + 72*gamma*noise*m1*m2**2 - 720*gamma*m1**3*m2**2 - 720*gamma*m1*m2**4 + 208*m2*q12 - 288*gamma*m2*q12 + 16*noise*m2*q12 - 40*gamma*noise*m2*q12 - 
        272*m1**2*m2*q12 + 240*gamma*m1**2*m2*q12 - 192*m2**3*q12 + 96*gamma*m2**3*q12 + 104*m1*q12**2 - 576*gamma*m1*q12**2 - 32*gamma*noise*m1*q12**2 + 336*gamma*m1**3*q12**2 + 1272*gamma*m1*m2**2*q12**2 + 96*m2*q12**3 - 480*gamma*m2*q12**3 - 
        168*gamma*m1*q12**4,
        # Var[q22]
        416 - 1648*gamma + 3032*gamma**2 + 48*noise - 272*gamma*noise + 608*gamma**2*noise + 32*gamma**2*noise**2 - 96*m1**2 + 256*gamma*m1**2 - 1504*gamma**2*m1**2 - 128*gamma**2*noise*m1**2 + 416*gamma**2*m1**4 - 416*m2**2 + 6432*gamma*m2**2 - 4384*gamma**2*m2**2 + 544*gamma*noise*m2**2 - 704*gamma**2*noise*m2**2 - 3392*gamma*m1**2*m2**2 + 
        832*gamma**2*m1**2*m2**2 + 320*m2**4 - 3392*gamma*m2**4 + 9056*gamma**2*m2**4 + 576*gamma**2*noise*m2**4 - 5760*gamma**2*m1**2*m2**4 - 5760*gamma**2*m2**6 - 768*m1*m2*q12 + 5056*gamma*m1*m2*q12 - 18112*gamma**2*m1*m2*q12 - 1088*gamma**2*noise*m1*m2*q12 + 
        6784*gamma**2*m1**3*m2*q12 - 4352*gamma*m1*m2**3*q12 + 18304*gamma**2*m1*m2**3*q12 + 784*q12**2 - 7008*gamma*q12**2 + 15776*gamma**2*q12**2 - 272*gamma*noise*q12**2 + 1168*gamma**2*noise*q12**2 + 1696*gamma*m1**2*q12**2 - 5504*gamma**2*m1**2*q12**2 - 320*m2**2*q12**2 + 
        6656*gamma*m2**2*q12**2 - 22784*gamma**2*m2**2*q12**2 - 576*gamma**2*noise*m2**2*q12**2 + 10112*gamma**2*m1**2*m2**2*q12**2 + 8640*gamma**2*m2**4*q12**2 + 2176*gamma*m1*m2*q12**3 - 15680*gamma**2*m1*m2*q12**3 + 80*q12**4 - 2480*gamma*q12**4 + 11576*gamma**2*q12**4 + 
        144*gamma**2*noise*q12**4 - 1440*gamma**2*m1**2*q12**4 - 4320*gamma**2*m2**2*q12**4 + 720*gamma**2*q12**6,
        # Cov[q22, m1]
        80*m1 + 52*gamma*m1 + 8*noise*m1 - 4*gamma*noise*m1 - 48*m1**3 - 64*gamma*m1**3 + 32*m1*m2**2 + 1016*gamma*m1*m2**2 + 72*gamma*noise*m1*m2**2 - 720*gamma*m1**3*m2**2 - 720*gamma*m1*m2**4 + 184*m2*q12 - 232*gamma*m2*q12 + 16*noise*m2*q12 - 32*gamma*noise*m2*q12 - 
        288*m1**2*m2*q12 + 240*gamma*m1**2*m2*q12 - 176*m2**3*q12 + 64*gamma*m2**3*q12 + 128*m1*q12**2 - 640*gamma*m1*q12**2 - 36*gamma*noise*m1*q12**2 + 360*gamma*m1**3*q12**2 + 1264*gamma*m1*m2**2*q12**2 + 88*m2*q12**3 - 440*gamma*m2*q12**3 - 
        180*gamma*m1*q12**4,
        # Cov[q22, m2]
        80*m1 + 52*gamma*m1 + 8*noise*m1 - 4*gamma*noise*m1 - 48*m1**3 - 64*gamma*m1**3 + 32*m1*m2**2 + 1016*gamma*m1*m2**2 + 72*gamma*noise*m1*m2**2 - 720*gamma*m1**3*m2**2 - 720*gamma*m1*m2**4 + 184*m2*q12 - 232*gamma*m2*q12 + 16*noise*m2*q12 - 32*gamma*noise*m2*q12 - 
        288*m1**2*m2*q12 + 240*gamma*m1**2*m2*q12 - 176*m2**3*q12 + 64*gamma*m2**3*q12 + 128*m1*q12**2 - 640*gamma*m1*q12**2 - 36*gamma*noise*m1*q12**2 + 360*gamma*m1**3*q12**2 + 1264*gamma*m1*m2**2*q12**2 + 88*m2*q12**3 - 440*gamma*m2*q12**3 - 
        180*gamma*m1*q12**4,
        # Var[m1]
        36 + 4*noise + 140*m1**2 + 8*noise*m1**2 - 96*m1**4 - 24*m2**2 - 96*m1**2*m2**2 - 40*m1*m2*q12 + 12*q12**2 + 48*m1**2*q12**2 + 20*m2**2*q12**2,
        # Cov[m1, m2]
        164*m1*m2 + 8*noise*m1*m2 - 96*m1**3*m2 - 96*m1*m2**3 + 36*q12 + 4*noise*q12 - 44*m1**2*q12 - 44*m2**2*q12 + 68*m1*m2*q12**2 + 12*q12**3,
        # Var[m2]
        36 + 4*noise - 24*m1**2 + 140*m2**2 + 8*noise*m2**2 - 96*m1**2*m2**2 - 96*m2**4 - 40*m1*m2*q12 + 12*q12**2 + 20*m1**2*q12**2 + 48*m2**2*q12**2
    ]))

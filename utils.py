from scipy.interpolate import splev,splprep
from scipy import optimize
import numpy as np

class Fit_airfoil():
    def __init__(self,airfoil,iLE=128):
        self.iLE = iLE
        self.tck, self.u  = splprep(airfoil.T,s=0)

        # parsec features
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0,1]
        ytelo = airfoil[-1,1]
        alphate, betate = self.get_te_angle()

        self.parsec_features = np.array([rle,xup,yup,yxxup,xlo,ylo,yxxlo,
                                         yteup,ytelo,alphate,betate]) 
        
        xaft, yaft, yxxaft = self.get_aftload()

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu,yu = splev(uLE, self.tck,der=1) 
        xuu,yuu = splev(uLE, self.tck,der=2)
        K = abs(xu*yuu-xuu*yu)/(xu**2+yu**2)**1.5
        return 1/K
    
    def get_up(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0,self.u[self.iLE]),method='bounded')
        uup = res.x
        xup ,yup = splev(uup, self.tck)

        xu,yu = splev(uup, self.tck, der=1)
        xuu,yuu = splev(uup, self.tck, der=2)

        yxx = (yuu*xu-xuu*yu)/xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(self.u[self.iLE],1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1)
        xuu,yuu = splev(ulo, self.tck, der=2)

        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

    def get_te_angle(self):
        xu,yu = splev(0, self.tck, der=1)
        yx = yu/xu
        alphate = np.arctan(yx)

        xu,yu = splev(1, self.tck, der=1)
        yx = yu/xu
        betate = np.arctan(yx)

        return alphate, betate
    

    def get_aftload(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0.75,1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1)
        xuu,yuu = splev(ulo, self.tck, der=2)

        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx
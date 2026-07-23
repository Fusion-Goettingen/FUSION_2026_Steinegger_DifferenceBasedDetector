import numpy as np, numba, scipy


@numba.njit(cache=True)
def sqrtm2x2(mat):
    #print(mat)
    mat = mat.ravel()
    assert(len(mat) == 4)
    a, c, c_, b = mat
    assert(c - c_ < 1e16)

    d =  np.sqrt(4 * np.abs(c)**2 + (a-b)**2)

    ev1 = (a + b - d) /2
    ev2 = (a + b + d) /2

    ev1 = np.sqrt(ev1) #func(ev1)
    ev2 = np.sqrt(ev2)

    A = ((a-b+d)*ev2 - (a-b-d)*ev1) /(2*d)
    B = ((b-a+d)*ev2 - (b-a-d)*ev1) /(2*d)
    C = c*(ev2-ev1)/d
    return np.array([[A, C],[C, B]])


# https://stackoverflow.com/questions/72742469/what-is-an-efficient-way-to-extract-eigenvalues-from-3x3-matrix-elements-stored

@numba.njit(cache=True)
def sqrtm(mat):
    val, vec = np.linalg.eigh(mat)
    #print(vec, val)
    return vec @ np.diag(np.sqrt(val)) @ np.linalg.inv(vec)


@numba.njit(cache=True)
def sqrtmvec(mats):
    arr = []
    for mat in mats:
        val, vec = np.linalg.eigh(mat) # mat is sym -> eigh 
        #print(vec, val)
        s = vec @ np.diag(np.sqrt(val)) @ np.linalg.inv(vec)
        arr.append(s)

    return arr



@numba.njit
def sqrtm3x3__(mat):
    func = np.sqrt
    # https://hal.science/hal-01501221/document
    mat = mat.ravel()
    assert(len(mat) == 9)
    a, d_, f_, d, b, e_, f, e, c = mat
    assert(d_ == d and f_ == f and e_ == e)

    x1 = a**2 + b**2 + c**2 - a*b -a*c -b*c + 3*(d**2 + f**2 + e**2)
    x2 = -(2*a-b-c)*(2*b-a-c)*(2*c-a-b) + \
         9*((2*c-a-b)*d**2 + (2*b-a-c)*f**2 + (2*a-b-c)*e**2) - 54*d*e*f

    # print(x1, x2, x1**2, x2**2)

    x13 = x1**3
    x22 = x2**2

    x1_sr = np.sqrt(x1)

    #print(4*x13-x22, x2)
    
    phi = np.atan(np.sqrt(4*x13-x22) / x2) if x2 != 0 else np.pi/2
    phi = phi + ( np.pi if x2 < 0 else 0)

    #print(phi)

    ev1 = (a+b+c-2*x1_sr* np.cos(phi/3))/3
    ev2 = (a+b+c+2*x1_sr*np.cos((phi-np.pi)/3))/3
    ev3 = (a+b+c+2*x1_sr*np.cos((phi+np.pi)/3))/3

    #print(ev1, ev2, ev3)

    m = lambda ev : (d*(c-ev)-e*f)/(f*(b-ev)-d*e)
    m1 = m(ev1)
    m2 = m(ev2)
    m3 = m(ev3)

    #print(m1, m2, m3)
    

    n = lambda m, ev: (1 + np.abs(m)**2 + np.abs(ev - c - e*m)**2)/f**2
    n1 = n(m1, ev1)
    n2 = n(m2, ev2)
    n3 = n(m3, ev3)

    ev1_ = func(ev1) / n1
    ev2_ = func(ev2) / n2
    ev3_ = func(ev3) / n3

    a_ = (ev1_ * np.abs(ev1-c-e*m1)**2 + ev2_ * np.abs(ev2-c-e*m2)**2 + ev3_ *np.abs(ev3-c-e*m3)**2) / f**2
    b_ = ev1_ * m1**2 + ev2_* m2**2 + ev3_ * m3**2
    c_ = ev1_ + ev2_ + ev3_
    d_ = (ev1_ * m1 * (ev1-c-e*m1) + ev2_ * m2 * (ev2-c-e*m2) + ev3_ * m3 * (ev3-c-e*m3) ) / f
    e_ = ev1_ * m1 + ev2_ * m2 + ev3_ * m3
    f_ = (ev1_ * (ev1-c-e*m1) + ev2_ * (ev2-c-e*m2) + ev3_ * (ev3-c-e*m3) ) / f

    return np.array([[a_, d_, f_],
                     [d_, b_, e_],
                     [f_, e_, c_]])


def test2x2():
    mat = np.array([[4, 2], [2, 6]])

    a = sqrtm2x2(mat)#, np.sqrt)
    a_ = sqrtm(mat)
    b = scipy.linalg.sqrtm(mat)

    print(a@a)
    print(a_@a_)
    print(b@b)


def test3x3():
    mat = np.array([[[4, -2, 0.001], 
                    [-2, 3, 1], 
                    [0.001, 1, 5]],
                    [[4, -2, 0.001], 
                    [-2, 3, 1], 
                    [0.001, 1, 5]]])

    a = sqrtm3x3vec(mat)#, np.sqrt)
    for b in a :
        print(b@b)
    #b = scipy.linalg.sqrtm(mat)
    #print(mat)
    #print(a@a)
    #print(b@b)


if __name__ == "__main__":
    test3x3()
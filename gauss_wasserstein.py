import scipy, numpy as np

import numba

from sqrtm import sqrtm2x2 as sqrtm2x2, sqrtm


def gauss_wasserstein_distance_(G1, G2):
    mu1, sigma1 = G1
    mu2, sigma2 = G2
    
    diff_mean = mu1 - mu2
    squared_diff_mean = np.dot(diff_mean, diff_mean) # euclidean norm

    sigma1_sqrt = sqrtm(sigma1)
    
    sigma_product = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    sigma_product_sqrt = scipy.linalg.sqrtm(sigma_product)
    trace_term = np.trace(sigma1 + sigma2 - 2 * sigma_product_sqrt)
    wasserstein_distance_sq = squared_diff_mean + trace_term
    
    return  wasserstein_distance_sq


@numba.njit(cache=True)
def gauss_wasserstein_distance_cho(G1, G2, bla):
    mu1, sigma1 = G1
    mu2, sigma2 = G2
    
    diff_mean = mu1 - mu2
    squared_diff_mean = np.dot(diff_mean, diff_mean) # euclidean norm


    #sigma1_sqrt = scipy.linalg.sqrtm(sigma1)
    sigma1_sqrt = np.linalg.cholesky(sigma1)
    
    sigma_product = sigma1_sqrt @ sigma2 @ sigma1_sqrt.T
    #sigma_product_sqrt = scipy.linalg.sqrtm(sigma_product)

    sigma_product_sqrt = sqrtm2x2(sigma_product)

    trace_term = np.trace(sigma1 + sigma2 - 2 * sigma_product_sqrt)
    wasserstein_distance_sq = squared_diff_mean + trace_term
    #wasserstein_distance = np.sqrt(wasserstein_distance_sq)
    
    return  wasserstein_distance_sq

def test():
    a = (np.array([1, 1]), np.diag([1, 1]))
    b = (np.array([0, 0]), np.diag([1, 1]))
    
    d = gauss_wasserstein_distance_(a, b)
    print(d)

    d = gauss_wasserstein_distance_cho(a, b)
    print(d)


if __name__ == "__main__":
    test()
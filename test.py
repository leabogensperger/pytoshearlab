import os
import cv2
import torch
import pytoShearLab2D
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ####################################################################################################################
    # load image and instantiate shearlet class
    image = torch.Tensor(cv2.imread(os.getcwd() + '/data/butterfly.jpg',0)[:300,500:800]/255.)
    M, N = image.shape
    assert M == N

    sl2D = pytoShearLab2D.ShearletSystem2D(useGPU=1, nScales=2, dims=[M,N], datatype=torch.complex128)
    sl2D_shearlets = sl2D.shearlets
    image = image.to(sl2D.device)[None,:,:]

    ####################################################################################################################
    # get shearlet coefficients
    coeffs = sl2D.apply(image)
    print('shearlet coefficients shape:', coeffs.shape)

    ####################################################################################################################
    # reconstruction with inverse shearlet transform
    image_rec = sl2D.apply_inv(coeffs)
    print('max. diff between input and recovered image: ', torch.max(torch.abs(image-image_rec)))
    
    ####################################################################################################################
    # test adjoint operator
    for _ in range(10):
        u = torch.randn_like(image_rec)
        v = torch.randn_like(coeffs)
        lhs = sl2D.apply(u).reshape(-1).dot(v.reshape(-1))
        rhs = sl2D.apply_transpose(v).reshape(-1).dot(u.reshape(-1))
        print(torch.max(torch.abs(lhs-rhs)))
from curses import color_pair
import os
import cv2
import torch
import numpy as np
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
    
    fig, ax = plt.subplots(1,3)
    for a in ax:
        a.axis('off')
    ax[0].imshow(image[0].cpu()), ax[0].set_title('image')
    ax[1].imshow(image_rec[0].cpu()), ax[1].set_title('reconstruction')
    ax[2].imshow(torch.abs(image-image_rec)[0].cpu()), ax[2].set_title('max. diff. %.8f' %(torch.max(torch.abs(image-image_rec)[0].cpu())))
    plt.show()

    ####################################################################################################################
    # test adjoint operator
    for _ in range(10):
        u = torch.randn_like(image_rec)
        v = torch.randn_like(coeffs)
        lhs = sl2D.apply(u).reshape(-1).dot(v.reshape(-1))
        rhs = sl2D.apply_transpose(v).reshape(-1).dot(u.reshape(-1))
        print('adjointness test residual:', torch.abs(lhs-rhs).item())
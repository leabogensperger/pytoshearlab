import torch
import torch.fft
import numpy as np
import torch.nn.functional as F

import utils.filter_tools as filter_tools
import utils.fft_tools as fft_tools

class ShearletSystem2D:
    def __init__(self, useGPU, nScales, dims, datatype=torch.complex64): 
        self.useGPU = useGPU
        if self.useGPU:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'
        self.datatype = datatype 

        self.nScales = nScales
        self.shearLevels = np.ceil(np.arange(1,nScales+1)/2).astype(int)

        self.nSh = self.get_nshearlets(self.shearLevels)
        self.cone_offs = int(self.nSh - 1)//2
        self.dims = dims

        self.h = self.set_h()
        self.P = self.get_directional_filter()

        # prepare other filters
        self.hj = self.scale_hj()
        self.gj = self.scale_gj(self.hj)
        self.W_high = self.get_bandpass() # bandpass commputed from g used to multiply on all shearlets
        self.lowpass = self.get_lowpass() # lowpass for last shearlet responsible for contrast

        # get shearlets
        self.shearlets = self.compute_shearlets()
        self.shearlets_norm = torch.sum(torch.pow(torch.abs(self.shearlets),2), axis=0)

    def set_h(self):        
        h = torch.Tensor(np.array([0.0104933261758410, -0.0263483047033631,
                                            -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                                            0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                                            0.0104933261758408])).to(self.device) # create h -> associated with scaling filter \phi (FIR Lowpass filter)
        return h

    def get_lowpass(self):
        lowpass = fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(
            filter_tools.padArray(torch.outer(self.hj[0], self.hj[0]), np.array([self.dims[0], self.dims[1]]))))) #fft_tools.fftshift
        return lowpass 

    def get_bandpass(self):
        # bandpass (extract highpass of W)
        W_high = torch.zeros((self.nScales,self.dims[0], self.dims[1]), dtype=self.datatype, device=self.device) 
        for j in range(len(self.gj)):
            W_high[j] = fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(filter_tools.padArray(self.gj[j], np.array([self.dims[0],self.dims[1]])))))
        return W_high 

    def scale_hj(self):
        hj = [None]*self.nScales
        hj[-1] = self.h
        PadW = hj[-1].shape[0] - 1 if hj[-1].shape[0] % 2 == 1 else hj[-1].shape[0]
        for j in range(self.nScales-2, -1, -1):
            hj[j] = F.conv1d(filter_tools.upsample(hj[j+1],1,1).reshape(1,1,-1), hj[-1].reshape(1,1,-1), padding=PadW)[0,0]
        return hj

    def scale_gj(self, hj):
        g = torch.pow(-1,torch.arange(hj[-1].shape[0], device=self.device))*hj[-1]
        gj = [None]*self.nScales
        gj[-1] = g

        PadW = hj[-1].shape[0] - 1 if hj[-1].shape[0] % 2 == 1 else hj[-1].shape[0]
        for j in range(self.nScales-2, -1, -1):
            gj[j] = F.conv1d(filter_tools.upsample(gj[j+1],1,1).reshape(1,1,-1), hj[-1].reshape(1,1,-1), padding=PadW)[0,0] 

        return gj

    def get_directional_filter(self):
        h0, _ = filter_tools.dfilters('dmaxflat4', 'd')/np.sqrt(2)
        P = filter_tools.modulate2(h0, 'c')
        P /= sum(sum(np.absolute(P))) 
        return torch.Tensor(P).to(self.device)

    def get_nshearlets(self, shearLevel): # determine number of resultant shearlets
        nSh = 1 # scale 0, contrast shearlet
        for i in range(len(shearLevel)):
            nSh += 2*(2*2**(shearLevel[i]) + 1) # full shearlet system (*2 for both cones)
        return nSh 

    def apply_transpose(self, coeff): # actual transpose of decomposition
        batchsize = coeff.shape[0]
        img = torch.zeros((batchsize, coeff.shape[1], coeff.shape[2]), dtype=torch.complex64, device=self.device) 
        img = (fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(coeff, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))*(self.shearlets[None,:])).sum(1)
        return torch.real(fft_tools.fftshift(torch.fft.ifftn(fft_tools.ifftshift(img, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1)))

    def apply(self, img):
        # input image img in (bs,m,n)
        # output coefficients (bs,ncoeff,m,n)
        batchsize = img.shape[0]
        coeffs = torch.zeros((batchsize, self.nSh, img.shape[-2], img.shape[-1]), dtype=self.datatype, device=img.device)
        img_freq = fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(img, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))
        coeffs = fft_tools.fftshift(torch.fft.ifftn(fft_tools.ifftshift(img_freq[:,None]*torch.conj(self.shearlets[None,:,:,:]), dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))
        return torch.real(coeffs)

    def apply_inv(self, coeff): # inverse op for synthesis
        batchsize = coeff.shape[0]
        img = torch.zeros((batchsize, coeff.shape[-2], coeff.shape[-1]), dtype=self.datatype, device=self.device) 
        img = (fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(coeff, dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))*self.shearlets[None,:,:,:]).sum(1)
        img = fft_tools.fftshift(torch.fft.ifftn(fft_tools.ifftshift(img/self.shearlets_norm[None,:], dim=(-2,-1)), dim=(-2,-1)), dim=(-2,-1))
        return torch.real(img)

    def compute_shearlets(self):
        shearlets = torch.zeros((self.nSh, self.dims[0], self.dims[1]), dtype=self.datatype, device=self.device) #
        self.scale_idx = torch.zeros((self.nSh))
        self.shear_idx = torch.zeros((self.nSh))
        shearlet_idx = 0

        for scale in np.arange(self.nScales):
            shearLevel = self.shearLevels[scale]
  
            # upsample directional filter
            P_up = filter_tools.upsample(self.P,0,np.power(2,shearLevel + 1)-1)

            # convolve P_up with lowpass -> remove high frequencies along vertical direction
            g_dyad_h = torch.outer(self.hj[len(self.hj)-shearLevel-1],self.gj[scale])
            padH, padW = g_dyad_h.shape[0] - 1, g_dyad_h.shape[1] -1
            psi_j0 = F.conv2d(P_up[None,None,:,:], g_dyad_h[None,None,:,:], padding=(padH,padW))[0,0]
            
            # psi_j0 = g_dyad_h.clone()
            psi_j0 = filter_tools.padArray(psi_j0,np.array([self.dims[0], self.dims[1]]))

            # upsample psi_j0
            psi_j0_up = filter_tools.upsample(psi_j0,1,np.power(2,shearLevel)-1)

            # convolve with lowpass
            lp_tmp = filter_tools.padArray(self.hj[len(self.hj)-max(shearLevel-1,0)-1][None,:], np.asarray(psi_j0_up.shape))
            lp_tmp_flip = torch.fliplr(lp_tmp)

            psi_j0_up = fft_tools.fftshift(torch.fft.ifftn(fft_tools.ifftshift(
                                    fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(lp_tmp)))*
                                    fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(psi_j0_up))))))

            for shearing in range(-np.power(2, shearLevel), np.power(2, shearLevel)+1):
                psi_j0_up_shear = filter_tools.dshear(psi_j0_up,shearing,1)

                # convolve with flipped lowpass
                psi_j0_up_shear = fft_tools.fftshift(torch.fft.ifftn(fft_tools.ifftshift(
                                            fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(lp_tmp_flip)))*
                                            fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(psi_j0_up_shear))))))

                shearlets[shearlet_idx] = fft_tools.fftshift(torch.fft.fftn(fft_tools.ifftshift(
                                        np.power(2,shearLevel)*
                                        psi_j0_up_shear[:,0:np.power(2,shearLevel)*self.dims[1]-1:np.power(2,shearLevel)]))) 

                shearlets[shearlet_idx + self.cone_offs] = shearlets[shearlet_idx].T

                self.scale_idx[shearlet_idx] = scale + 1
                self.scale_idx[shearlet_idx + self.cone_offs] = scale + 1

                self.shear_idx[shearlet_idx] = shearing
                self.shear_idx[shearlet_idx + self.cone_offs] = shearing

                shearlet_idx += 1

        shearlets[-1] = self.lowpass
        self.scale_idx[-1] = 1.
        
        return shearlets
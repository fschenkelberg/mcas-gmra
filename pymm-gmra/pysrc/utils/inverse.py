from pysrc.utils.utils import *
import numpy as np

#This file contains functions for computing the inverse of a tree

# Transcription of FGWT from matlab. We need this to compute CelWavCoeffs for the inverse
def fgwt(wavelet_tree, X):
    J_max = depth(wavelet_tree.root) #max scales not depth
    leafs = get_leafs(wavelet_tree.root)
    CelWavCoeffs = np.zeros((X.shape[0],J_max)) 
    CelScalCoeffs = np.zeros((X.shape[0],J_max))

    for leaf in leafs: #TODO
        #the index of the single point in this leaf
        curIdx = leaf.idxs[0]
        #the indices at the parent of the leaf
        p = path(leaf)
        iFineNet = leaf #something based on path
        iCoarseNet = iFineNet.parent
        pt_idxs = leaf.idxs
        j = level(leaf)
        # if j==1:
        #     CelWavCoeffs[curIdx - 1][0] = leaf.parent.wav_basis.dot(X[:, pt_idxs] - leaf.parent.center)
        #     CelScalCoeffs[curIdx - 1][0] = CelWavCoeffs[curIdx - 1][0]
        Projections_jmax = X[pt_idxs]
        
        if leaf.wav_basis is not None:
            CelScalCoeffs[j] = leaf.wav_basis.dot(X[:, pt_idxs] - leaf.center)
            CelWavCoeffs[curIdx][j] = ComputeWaveletCoeffcients(
                CelScalCoeffs[curIdx][j],
                leaf.basis,
                leaf.wav_basis)
        for node in reversed(p[:-1]):
            j= level(node)
            iFinerNet= iFineNet
            iFineNet = iCoarseNet
            iCoarseNet = iFineNet.parent
            #TODO: what is ScalBasisChange?
            if node.wav_basis is not None:
                CelScalCoeffs[curIdx][j] = (
                            iFineNet.wav_basis.dot(iFinerNet.center - iFineNet.center))
            if j==1 or iCoarseNet.parent is None:
                break
            if iFineNet.parent.wav_basis is not None and CelScalCoeffs[curIdx][j] is not None:
                CelWavCoeffs[curIdx][j] = ComputeWaveletCoeffcients(
                            CelScalCoeffs[curIdx][j],
                            iFineNet.basis,
                            iFineNet.wav_basis)
            CelWavCoeffs[curIdx][0] = CelScalCoeffs[curIdx][0]

    return CelWavCoeffs



#A Helper function for fgwt
def ComputeWaveletCoeffcients(data_coeffs, scalBases, wavBases):
    wavCoeffs = wavBases.dot((scalBases.T.dot(data_coeffs)))

    return wavCoeffs

# This is a transcription of IGWT from matlab, it outputs the inverse
def reconstruct_X(tree, X, CelWavCoeffs):

    #invert along path to root
    #result will be orig dim x n_pts x scale
    J_max = depth(tree.root)
    Projections = np.zeros((X.shape[1], X.shape[0], J_max))
    for leaf in get_leafs(tree.root):
        pt_idx = leaf.idxs
        x_matj = np.zeros((X.shape[1],len(pt_idx),J_max))
        chain = path(leaf)
        for j in reversed(range(J_max)):
            node = chain[j]
            if node.wav_consts is not None:
                x_tmp = node.wav_consts
            else:
                x_tmp = None
            if node.wav_basis is not None and CelWavCoeffs[pt_idx][j] is not None:
                x_tmp = node.wav_basis.T.dot(CelWavCoeffs[pt_idx][j]) + x_tmp
            if x_tmp is not None and x_tmp.shape[1]==x_matj.shape[1]:
                x_matj[:,:,j] = x_tmp

        Projections[:,pt_idx,:] = np.cumsum(x_matj,2).reshape(Projections[:,pt_idx,:].shape)

    return Projections

# Given a wavelettree and input matrix X, computes the inverse from the tree at every scale 
# Output will be a matrix with dim (# dimensions of original data, # pts, # scales)
def invert(wavelet_tree, X):
    #This populates wav_basis and centers variables within the tree
    wavelet_tree.make_wavelets(X)
    #This computes CelWavCoeffs
    CelWavcoeffs = fgwt(wavelet_tree, X)
    #Taking the actual inverse after we have all prereq computations
    projections = reconstruct_X(wavelet_tree, X, CelWavcoeffs)
    return projections
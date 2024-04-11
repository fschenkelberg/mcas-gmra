from pysrc.utils.utils import *
import numpy as np
import numpy.matlib

#This file contains functions for computing the inverse of a tree

# Transcription of FGWT from matlab. We need this to compute CelWavCoeffs for the inverse
# Line number comments refer to the line number in the original FGWT function
def fgwt(wavelet_tree, X):
    X = X.T
    print(X.shape)
    J_max = depth(wavelet_tree.root) #max scales not depth
    leafs = get_leafs(wavelet_tree.root)
    CelWavCoeffs = [[None]*(J_max+1) for i in range(X.shape[1])] #np.zeros((X.shape[0],J_max+1)) 
    CelScalCoeffs = [[None]*(J_max+1) for i in range(X.shape[1])] #np.zeros((X.shape[0],J_max+1))
    CelTangCoeffs = [[None]*(J_max+1) for i in range(X.shape[1])]
    
    for leaf in leafs:
        #the index of the single point in this leaf
        curIdx = leaf.idxs[0]
        #the indices at the parent of the leaf
        p = path(leaf)
        iFineNet = leaf #something based on path
        iCoarseNet = iFineNet.parent
        pt_idxs = leaf.idxs
        j = level(leaf)

        #If this is the root
        if j==1: # Line 136
            CelWavCoeffs[curIdx][j] = leaf.wav_basis.dot(X[:, pt_idxs] - leaf.center)
            CelScalCoeffs[curIdx][j] = CelWavCoeffs[curIdx][1]
            leaf.CelWavCoeffs[curIdx] = CelWavCoeffs[curIdx][j]
        else: # Line 142
            CelScalCoeffs[curIdx][j] = iFineNet.basis.dot(X[:,pt_idxs]- iFineNet.center)
            
            Projections_jmax = X[:,pt_idxs]
        
            if iFineNet.wav_basis is not None: # Line 151
                # CelScalCoeffs[curIdx][j] = leaf.wav_basis.dot(X[:, pt_idxs] - leaf.center)
                CelWavCoeffs[curIdx][j] = ComputeWaveletCoeffcients(
                    CelScalCoeffs[curIdx][j],
                    iFineNet.basis,
                    iFineNet.wav_basis)
                leaf.CelWavCoeffs[curIdx] = CelWavCoeffs[curIdx][j]
            CelTangCoeffs[curIdx][j] = np.zeros((iCoarseNet.basis.shape[0],len(leaf.idxs)))

        for node in reversed(p[:-1]): # Line 179
            j = level(node)
            iFinerNet = iFineNet
            iFineNet = node
            iCoarseNet = node.parent
            #TODO: what is ScalBasisChange? Also CelTangCoeffs
            if node.basis is not None: # Line 199
                CelScalCoeffs[curIdx][j] = numpy.matlib.repmat( 
                    iFineNet.basis.dot(iFinerNet.center - iFineNet.center),1,len(iFinerNet.idxs))
            if j==1 or iCoarseNet.parent is None: # Line 203
                break
            if iFineNet.wav_basis is not None and CelScalCoeffs[curIdx][j] is not None: # Line 205
                CelWavCoeffs[curIdx][j] = ComputeWaveletCoeffcients(
                            CelScalCoeffs[curIdx][j],
                            iFineNet.basis,
                            iFineNet.wav_basis)
                iFineNet.CelWavCoeffs[curIdx] = CelWavCoeffs[curIdx][j]
            CelWavCoeffs[curIdx][1] = CelScalCoeffs[curIdx][1] # Line 236

    return CelWavCoeffs



#A Helper function for fgwt
def ComputeWaveletCoeffcients(data_coeffs, scalBases, wavBases):
    # print(wavBases.shape, scalBases.T.shape, data_coeffs.shape)
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
        # TODO: Check Chain Generation
        chain = path(leaf)
        # TODO: Verify Index Calculation
        for j in reversed(range(len(chain))):
            # Debugging info
            # print(f"Index j: {j}, Length of chain: {len(chain)}")
            node = chain[j]
            if node.wav_consts is not None:
                x_tmp = node.wav_consts
            else:
                x_tmp = None
            # if node.wav_basis is not None and CelWavCoeffs[pt_idx][j] is not None:
            #     x_tmp = node.wav_basis.T.dot(CelWavCoeffs[pt_idx][j]) + x_tmp
            if node.wav_basis is not None and pt_idx in node.CelWavCoeffs:
                x_tmp = node.wav_basis.T.dot(node.CelWavCoeffs[pt_idx]) + x_tmp
            if x_tmp is not None and x_tmp.shape[1]==x_matj.shape[1]:
                x_matj[:,:,j] = x_tmp

        Projections[:,pt_idx,:] = np.cumsum(x_matj,2).reshape(Projections[:,pt_idx,:].shape)

    return Projections

# Given a wavelettree and input matrix X, computes the inverse from the tree at every scale 
# Output will be a matrix with dim (# dimensions of original data, # pts, # scales)
def invert(wavelet_tree, X):
    #This populates wav_basis and centers variables within the tree
    wavelet_tree.make_wavelets(X)

    #check counts of populated wav vars
    # print(check_wav_vars(wavelet_tree.root))
    # print(wavelet_tree.num_nodes)
    # print(len(get_leafs(wavelet_tree.root)))

    #This computes CelWavCoeffs
    CelWavcoeffs = fgwt(wavelet_tree, X)
    #Taking the actual inverse after we have all prereq computations
    projections = reconstruct_X(wavelet_tree, X, CelWavcoeffs)
    return projections
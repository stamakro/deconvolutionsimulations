import numpy as np
import matlab.engine
from scipy.stats import pearsonr

# generate an atlas of some tissues and markers
Ntissues = 5
Ngenes = 50


mus = np.random.normal(0, 1, size=(Ntissues, Ngenes))


# mix the tissues at specific proportions
proportions = np.arange(Ntissues + 1)[1:]
proportions = proportions / np.sum(proportions)

print('True proportions:')
print(proportions)

mixture = mus.T.dot(proportions)

# convert cfDNA profile and atlas to matlab matrices
cfDNA = matlab.double(mixture)
atlas = matlab.double(mus)

C = 1.0
eps = 0.1

# init matlab
eng = matlab.engine.start_matlab()
eng.cd(r'/home/stavros/Desktop/code/deconvolutionsimulations/src/matlab/', nargout=0)

# call matlab implementation of deconvolution
matlabres = eng.deconvolve(cfDNA, atlas, C, eps, nargout=1)

estProportions = np.array(matlabres).reshape(-1,)
print('Estimated proportions:')
print(estProportions)

print('Correlation true vs estimated:')
print(pearsonr(estProportions, proportions))

import numpy as np
import matlab.engine
from scipy.stats import pearsonr

# generate an atlas of some tissues and markers
Ntissues = 2
Ngenes = 3
N = 10


#mus = np.random.normal(0, 1, size=(Ntissues, Ngenes))
mus = np.array([[1,1,1], [-1,-0.5,0.1]])

# mix the tissues at specific proportions
fracs = np.random.rand(N)
proportions = np.vstack((fracs, 1-fracs))


print('True proportions:')
print(proportions)

mixture = mus.T.dot(proportions).T
mixture = matlab.double(mixture)

# convert cfDNA profile and atlas to matlab matrices

musFake = np.array([[0.7,1.3,1.8], [-1.1,-0.3,0.12]])

atlas = matlab.double(musFake)

C = 1.0
eps = 0.1


estimatedCorrect = np.zeros(N)

for i in range(N):
    cfDNA = mixture[i]

    # init matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/stavros/Desktop/code/deconvolutionsimulations/src/matlab/', nargout=0)

    # call matlab implementation of deconvolution
    matlabres = eng.deconvolve(cfDNA, atlas, C, eps, nargout=1)

    estimatedCorrect[i] = np.array(matlabres).reshape(-1,)[0]

print('Estimated proportions:')
print(estimatedCorrect)

print('Correlation true vs estimated:')
print(pearsonr(estimatedCorrect, proportions[0]))
print('MAD true vs estimated:')
print(np.median(np.abs(estimatedCorrect-proportions[0])))

# fine-tune first column, holding second fixed
for i in range(Ngenes):
    print('gene %d, true value: %f' % (i, mus[0,i]))
    print('gene %d, current value: %f' % (i, musFake[0,i]))

    values = np.array(mixture)[:,i]
    ff = proportions[0]

    corr = np.mean((values - (1-ff)*musFake[1,i]) / ff)

    print('gene %d, corrected value: %f' % (i, corr))
    musFake[0,i] = corr

estimated2 = np.zeros(N)
atlas = matlab.double(musFake)

for i in range(N):
    cfDNA = mixture[i]

    # init matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/stavros/Desktop/code/deconvolutionsimulations/src/matlab/', nargout=0)

    # call matlab implementation of deconvolution
    matlabres = eng.deconvolve(cfDNA, atlas, C, eps, nargout=1)

    estimated2[i] = np.array(matlabres).reshape(-1,)[0]

print('Correlation true vs estimated:')
print(pearsonr(estimated2, proportions[0]))
print('MAD true vs estimated:')
print(np.median(np.abs(estimated2-proportions[0])))

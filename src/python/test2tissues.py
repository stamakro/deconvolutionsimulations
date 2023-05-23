import numpy as np
from scipy.stats import nbinom, pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from sklearn.svm import LinearSVR


class SimpleDeconvolver():
    def __init__(self, means, a0=None):
        self.estimatedMus = means
        self.Ntissues = self.estimatedMus.shape[0]
        self.hessian = self.estimatedMus.dot(self.estimatedMus.T)

        if a0 is not None:
            self.initial = a0
        else:
            self.initial = np.ones(self.Ntissues) / self.Ntissues

        self.constraints = self.getConstraints()



    def fit(self, x):
        res = minimize(SimpleDeconvolver.loss, self.initial, args=(x, self.estimatedMus), method='trust-constr', jac=True, hess=self.getHessian, constraints=self.constraints,)

        if res['success']:
            return res['x']


    def getConstraints(self):

        A = np.eye(self.Ntissues)

        a2 = np.ones(self.Ntissues)

        A = np.vstack((A, a2))

        lb = np.concatenate((np.zeros(self.Ntissues), [1]))
        ub = np.concatenate((np.ones(self.Ntissues), [1]))

        return LinearConstraint(A, lb, ub)

    def getHessian(self, *args):
        return self.hessian


    @staticmethod
    def loss(a, x, estimatedMus):

        diffs = x - estimatedMus.T.dot(a)

        loss = np.mean(diffs ** 2) / 2.

        jac = - np.mean(diffs * estimatedMus,1)

        return loss, jac



def getMeanVariance(logmu, size):
    mu = np.exp(logmu)
    var = mu + (mu ** 2) / size

    return mu, var

def getNp(mu, var):
    p = mu / var
    n = (mu ** 2) / (var - mu)

    return n, p


np.random.seed(100)
Ntissues = 20
NtissueSamples = 5
Ngenes = 50

snoise = 1.

cc = []
ccs = []

for Ntissues in range(3,31):


    mus = np.random.normal(0, 1, size=(Ntissues, Ngenes))
    sigmas = np.random.exponential(0.5, size=(Ntissues, Ngenes))

    referenceData = np.zeros((NtissueSamples * Ntissues, Ngenes))
    labels = np.zeros(NtissueSamples * Ntissues, int)
    estimatedMus = np.zeros(mus.shape)
    estimatedSigmas = np.zeros(mus.shape)

    ss = 0
    for i in range(Ntissues):
        referenceData[ss:ss+NtissueSamples] = np.random.normal(mus[i], sigmas[i], size=(NtissueSamples, Ngenes))

        labels[ss:ss+NtissueSamples] = i

        estimatedMus[i] = np.mean(referenceData[ss:ss+NtissueSamples], axis=0)
        estimatedSigmas[i] = np.std(referenceData[ss:ss+NtissueSamples], ddof=1, axis=0)


        ss += NtissueSamples


    Nsamples = 200
    if Ntissues == 2:
        fraction1 = np.random.rand(Nsamples)
        trueFractions = np.vstack((fraction1, 1-fraction1)).T
    else:
        trueFractions = np.random.dirichlet(np.ones(Ntissues) / Ntissues, Nsamples)


    data = trueFractions.dot(np.random.normal(mus,sigmas))
    if snoise > 0:
        data += np.random.normal(0, snoise, size=data.shape)


    pred = np.zeros((data.shape[0], Ntissues))
    for i in range(data.shape[0]):
        reg = LinearSVR(fit_intercept=False)

        reg.fit(estimatedMus.T, data[i])

        tmpFrac = np.maximum(reg.coef_, 0)

        pred[i] =  tmpFrac / np.sum(tmpFrac)


    corrs = np.zeros(Nsamples)
    for i in range(corrs.shape[0]):
        corrs[i] = pearsonr(pred[i], trueFractions[i])[0]


    cc.append(corrs)
    # ccs.append(np.std(corrs, ddof=1))

# print(pearsonr(pred[:,1], trueFractions[:,1]))
# print('')
# print(spearmanr(pred[:,1], trueFractions[:,1]))
#
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.scatter(np.arange(2,30), cc, edgecolor='k')
# ax.errorbar(np.arange(2,30), cc, yerr=ccs, fmt='none')
ax.boxplot(cc, whis=(1,99))

ax.set_xticklabels([str(k) for k in range(3,31)])
ax.set_xlabel('#tissues')
ax.set_ylabel('correlation predicted vs true fractions')
ax.set_title('N = 200')

ax.set_ylim(0,1.1)
plt.grid(which='both', axis='y')
# fig.savefig('easy.png')


print('#Ref / ct: %d' % NtissueSamples)
print('noise in y: %f' % snoise)
print('CpG sites: %d' % Ngenes)





plt.show()













# end

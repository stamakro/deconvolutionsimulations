import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, dirichlet, pearsonr, spearmanr, gaussian_kde
import pickle
#import statsmodels.api as sm
from copy import deepcopy
import sys
from scipy.spatial.distance import cdist
import matlab.engine

def geneLength(region: str) -> int:
    # takes a region, e.g. 'chr3:1000-1234' in this format, and returns the length of the region

    start, end = region.split(':')[1].split('-')

    return int(end) - int(start)


def splitAtlas(sampleNames: np.ndarray, celltypes: np.ndarray, seed: int) -> list:
    # remove one random sample from each CT for leave-one-out

    # immune cells together
    # neurons/oligodendrocytes together
    # colon independent
    # melanoma also independent
    # if cell types are added, this has to change
    np.random.seed(seed)

    allCellTypes = np.sort(np.unique(celltypes))
    testSamples = []

    # randomly shuffle all colon samples and pick the last one in this random order
    colonRnd = np.random.permutation(np.where(celltypes == 'colon')[0])
    if len(colonRnd) > 1:
        testSamples = [sampleNames[colonRnd[-1]]]

    # similar for melanoma
    melaRnd = np.random.permutation(np.where(celltypes == 'melanoma')[0])
    if len(melaRnd) > 1:
        testSamples += [sampleNames[melaRnd[-1]]]

    # for neurons and oligos, take the sample of the same individual
    # so pick a number from 1-11
    brainRnd = np.random.randint(1,12)
    if '%d-N' % brainRnd in sampleNames:
        testSamples += ['%d-N' % brainRnd]

    if '%d-O' % brainRnd in sampleNames:
        testSamples += ['%d-O' % brainRnd]

    # ignore these old samples
    # tissueNames = np.random.permutation(['S13-095', 'S14-073', 'S14-074', 'S14-129'])
    #
    # if ('Liver-%s' % tissueNames[-1]) in sampleNames:
    #     testSamples += ['Liver-%s' % tissueNames[-1]]
    #
    # if ('Ovary-%s' % tissueNames[-1]) in sampleNames:
    #     testSamples += ['Ovary-%s' % tissueNames[-1]]
    #
    # if ('Thyroid-%s' % tissueNames[-1]) in sampleNames:
    #     testSamples += ['Thyroid-%s' % tissueNames[-1]]

    # this is similar for the immune cells, take all 4 from the same individual
    # (one Neutrophile sample is missing)
    # first individual is A3832-A3835, next starts from A3836 etc
    immuneNumbers = np.random.permutation(np.arange(32,69,4))

    # B-cell
    testSamples.append('A38%d' % immuneNumbers[-1])
    # T-cell
    testSamples.append('A38%d' % (immuneNumbers[-1]+1))
    # Monocyte
    testSamples.append('A38%d' % (immuneNumbers[-1]+2))
    neutroS = 'A38%d' % (immuneNumbers[-1]+3)
    if neutroS in sampleNames:
        testSamples.append(neutroS)

    return testSamples


def findMarkersTstat(datasource: dict, minTstat: float, maxMark=None, NmarkersPerCellType=500):
    ''' for each region, compare its methylation value in one ct to all others. if the t-statisitc is
    at least minTstat in all comparisons, then this is selected as a marker for that cell type.
    This is done separately for hyper- and hypo-methylated markers

    datasource: (dict) a dictionary with the cell types as keys and a dataframe of the methylation values
                of the corresponding samples as values
    minTstat:   (float) the minimum value of t-statistic required for a region to be selected as a marker
    maxMark:    (int) ignore, leave it none
    NmarkersPerCellType: (int) cap the number of markers selected per cell type

    '''

    # calculate median, min, max per marker in each ct

    # gather all available cell types
    allCellTypes = np.sort(list(datasource.keys()))

    # number of regions, here we assume that all cell types have the same regions in the same order!
    Ngenes = datasource[allCellTypes[0]].shape[1]

    # compare all cell types against all others, for all regions
    tstats = np.zeros((len(allCellTypes), len(allCellTypes), Ngenes))
    for i, ct1 in enumerate(allCellTypes):
        x1 = datasource[ct1]
        for j, ct2 in enumerate(allCellTypes):
            # i don't hvae to compare ct1 to ct1, ttest will be 0
            if i != j:
                x2 = datasource[ct2]
                # now x1 and x2 are 2 DFs with the same number of columns (regions)
                # we can thus do all t-tests at the same time
                tstats[i,j], _ = ttest_ind(x1, x2)

    # if one cell type is constant (usually all 0's) the variance is undefined and thus also the t-statistic becomes NaN
    # I set these to 0 now, but this might not be ideal
    # if ct1 is always 0 but other cts not, this might be a good hypomethylated marker
    tstats[np.isnan(tstats)] = 0.

    # hypermethylated markers using mean + t statistic
    # this dictionary has as keys the cell types and as values a list of
    # indices of the hypermethylated markers found for this ct
    markersMeanHyper = dict()
    print('hypermethylated, mean')
    for i, ct in enumerate(allCellTypes):
        # to find markers for ct i, we have to look at all t-statistics between i and the rest (excl. i)
        remaining = np.setdiff1d(np.arange(len(allCellTypes)), [i])
        subMatrix = tstats[i, remaining]

        # say you have 3 cts and do two t-tests for ct 1
        # ct1-ct2: t = 0.5, ct1-ct3: t = 5
        # this is a good marker for discriminating ct1/3 but not ct1-2
        # so overall it's not a good marker for ct 1 and gets the smaller score of the two, 0.5
        # this is what this line below does
        fcToMostSimilar = np.min(subMatrix, axis=0)

        # count how many markers for ct i are above the minTstat value
        # but if this number is larger than NmarkersPerCellType, then we cap the number of
        # selected markers to NmarkersPerCellType
        tmpMarkerCount = np.minimum(NmarkersPerCellType, np.sum(fcToMostSimilar > minTstat))
        print(ct, tmpMarkerCount)

        # sort all min t-stats for ct i from smallest to largest and
        # select the top tmpMarkerCount and store their indices in the DF
        selectedMarkers = np.argsort(fcToMostSimilar)[-tmpMarkerCount:]

        # the selected markers are still from worst to best marker,
        # so flip their order and put them in the dictionary of markers
        markersMeanHyper[ct] = selectedMarkers[::-1]

    # it's not good to have 10 markers for ct 1 and 500 markers for ct 2
    # the method works better if there is more or less similar number of markers per ct
    # this code limits the number of selected markers per ct to maxMark
    # or if maxMark is less than the minimum, you can limit it further
    if maxMark is None:
        mm = min([len(v) for k,v in markersMeanHyper.items()])
    else:
        mm = maxMark
    print('Will select at most %d markers' % mm)
    bigSet = set()
    tt = 0
    for ct in markersMeanHyper:
        bigSet = bigSet.union(set(markersMeanHyper[ct][:mm]))
        markersMeanHyper[ct] = markersMeanHyper[ct][:mm]

    if len(bigSet) != mm * len(allCellTypes):
        print('At least one site is marker for multiple cell types')

    # hypomethylated markers using mean + t statistic
    # exactly the same story as above but for hypomethylated markers
    markersMeanHypo = dict()
    print('hypomethylated, mean')
    for i, ct in enumerate(allCellTypes):
        remaining = np.setdiff1d(np.arange(len(allCellTypes)), [i])

        subMatrix = tstats[i, remaining]
        fcToMostSimilar = np.max(subMatrix, axis=0)

        tmpMarkerCount = np.minimum(NmarkersPerCellType, np.sum(fcToMostSimilar < -minTstat))
        print(ct, tmpMarkerCount)

        selectedMarkers = np.argsort(fcToMostSimilar)[:tmpMarkerCount]

        markersMeanHypo[ct] = selectedMarkers


    if maxMark is None:
        mm = min([len(v) for k,v in markersMeanHypo.items()])
    else:
        mm = maxMark

    print('Will select at most %d markers' % mm)
    bigSet = set()
    tt = 0
    for ct in markersMeanHyper:
        bigSet = bigSet.union(set(markersMeanHypo[ct][:mm]))
        markersMeanHypo[ct] = markersMeanHypo[ct][:mm]

    if len(bigSet) != mm * len(allCellTypes):
        print('At least one site is marker for multiple cell types')

    # put all together in one dictionary
    allMarkers = {'hyper': markersMeanHyper, 'hypo': markersMeanHypo}

    return allMarkers

def findMarkersTstatPairwise(datasource: dict, minTstat: float, maxMark: int, minMark: int, verbose=True) -> list:
    ''' this routine finds at least minMark and at most maxMark markers for each pair of cell types

    datasource: (dict) a dictionary with the cell types as keys and a dataframe of the methylation values
                of the corresponding samples as values
    minTstat:   (float) the minimum value of t-statistic required for a region to be selected as a marker
    minMark:    (int) the minimum number of markers to select for each pair of two cell types
    maxMark:    (int) the maximum number of markers to select for each pair of two cell types
    verbose:    (bool) whether to print updates about all comparisons

    returns a list of indices to all the markers
    '''
    assert minMark < maxMark
    assert minTstat >= 0.

    # list of all cell types
    allCellTypes = np.sort(list(datasource.keys()))

    # nr of features (CpG islands/clusters etc)
    Ngenes = datasource[allCellTypes[0]].shape[1]

    allMarkers = set()

    for i, ct1 in enumerate(allCellTypes[:-1]):
        x1 = datasource[ct1]
        for j, ct2 in enumerate(allCellTypes[(i+1):], i+1):
            # do all pairs of t-tests between each 2 cell types
            x2 = datasource[ct2]
            # at this point x1 has all the samples of ct1 and x2 all the samples of ct2
            # do the t-test, removing constant regions, where t statisitc is NaN
            tstats, _ = ttest_ind(x1, x2)
            tstats[np.isnan(tstats)] = 0.

            # find the indices of the sorted t-stats
            ii = np.argsort(tstats)
            # see how many hypermethylated sites for ct 1 exceed minTstat
            potentialHyper = np.sum(tstats > minTstat)
            if potentialHyper > minMark:
                # if it's more than minMark, we take them all unless they are more than maxMark
                mm = np.minimum(potentialHyper, maxMark)
                hyper = ii[-mm:]

            else:
                # if it's less than minMark, select the top minMark
                hyper = ii[-minMark:]

            # exactly the same for hypomethylated in ct1
            potentialHypo = np.sum(tstats < -minTstat)
            if potentialHypo > minMark:
                mm = np.minimum(potentialHypo, maxMark)
                hypo = ii[:mm]

            else:
                hypo = ii[:minMark]

            if verbose:
                print('%s vs %s' % (ct1, ct2))
                print('Found %d hypermethylated, %d hypomethylated' % (len(hyper), len(hypo)))

            # add indices of new markers to the set of existing markers
            for marker in hyper:
                allMarkers.add(marker)
            for marker in hypo:
                allMarkers.add(marker)

            if verbose:
                print('Currently %d markers in total' % len(allMarkers))

    allMarkers = sorted(list(allMarkers))
    return allMarkers


def renameSample(names: list) -> dict:
    '''
    fix the problem where some samples are number 1-N vs neuron_1
    names:  list, a list of all sample names

    RETURNS a dictionary which can be used to fix the names by DataFrame.rename
    '''
    
    renameDict = dict()
    for name in names:
        if 'neuron' in name:
            # from neuron_X to X-N
            number = int(name.split('_')[1])
            renameDict[name] = str(number) + '-' + 'N'

        elif 'oligodendrocyte' in name:
            number = int(name.split('_')[1])
            renameDict[name] = str(number) + '-' + 'O'

    return renameDict


# magic starts here! confidence
if __name__ == '__main__':

    # file with sample info, cell type etc
    sampleInfoPath = '../../data/atlas-samples-overview.csv'
    # folder where methylation counts are stored
    outputPath = '../../data/'

    # remove sex chromosomes
    acceptedChromosomes = set(['chr' + str(i) for i in range(1,23)])

    # read methylation data
    data = pd.read_csv(outputPath + 'counts_aggregated_clusters.csv', index_col=0).T

    # read sample info, be careful if it is split by comma or semi-colon
    sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=';')
    if sampleInfo.shape[1] == 0:
        sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=',')

    assert sampleInfo.shape[0] == data.shape[0]

    # fix names of neurons and ologodendrocytes in sample info
    rd = renameSample(sampleInfo.index)
    sampleInfo.rename(index=rd, inplace=True)


    # make sure that the sample info and methylation counts are in the same order
    sampleInfo = sampleInfo.loc[data.index]
    assert (sampleInfo.index == data.index).all()

    assert sampleInfo.shape[0] == data.shape[0]

    # concatenate the two into a big dataframe
    bigDF = pd.concat((data, sampleInfo), axis=1, ignore_index=False)

    print('Samples before filtering: %d' % bigDF.shape[0])

    # filtering low quality samples
    bigDF = bigDF[bigDF['CpG/Total'] > 20.]
    bigDF = bigDF[bigDF['Used reads'] > 3000000]

    print('Samples after filtering: %d' % bigDF.shape[0])

    print('Samples after filtering per cell type:')
    print(bigDF.groupby('Cell type').count().iloc[:,0])

    # keep track of how many samples with have per cell type
    cellTypePresenceDraft = bigDF.groupby('Cell type').count().iloc[:,0].to_dict()


    # normalize the data by region length and library size
    # log TPM normalization

    # put all methylation counts into an array
    sampleCountsAutosomal = bigDF[bigDF.columns[pd.Series(bigDF.columns).apply(lambda x: x.split(':')[0]).isin(acceptedChromosomes)]]
    countsAutosomal = np.array(sampleCountsAutosomal, int)

    # calculate length of each region
    lengths = np.array(pd.Series(sampleCountsAutosomal.columns).apply(geneLength))
    # calculate methylation count per unit length
    countsPerBase = countsAutosomal / lengths
    total = np.sum(countsPerBase,1)

    # make sure each sample adds to 1million by dividing by the sample's total and multiplying by 1million
    tpm = 1e6 * (countsPerBase.T / total).T

    # log transform to make log-TPM, with pseudo-count of 1 to avoid log(0)
    logtpm = np.log(tpm + 1)
    print(logtpm.shape)
    assert not np.sum(np.isnan(logtpm))

    # put back into a pandas DF
    normalizedData = pd.DataFrame(data=logtpm, index=sampleCountsAutosomal.index, columns=sampleCountsAutosomal.columns)


    # to keep track of how many samples per ct, but now we need it in a list-like format, not as a dict
    allCellTypes = sorted(bigDF['Cell type'].unique())
    celltypePresence = np.zeros(len(allCellTypes))
    for i,ct in enumerate(allCellTypes):
        celltypePresence[i] = cellTypePresenceDraft[ct]

    # keep all methylation profiles of the same cell type in the same separate DataFrame
    # put all together in a dictionary indexed by the cell type name
    # this will make life easier later
    vstPerCt = dict()
    for ct in allCellTypes:
        ii = np.where(bigDF['Cell type'] == ct)[0]

        vstPerCt[ct] = normalizedData.iloc[ii]


    # NmarkersPerCellType = 500

    # minimum (absolute) value of t statistic required for a region to be considered a marker
    minTstat = 1.5

    # this routine finds all the markers
    allMarkers = findMarkersTstat(vstPerCt, minTstat)


    groups = bigDF.groupby('Cell type').groups

    # re-order the samples so that samples of the same cell type are all next to each other
    # this will help us draw a heatmap of the markers
    newInd = []
    for ct in allCellTypes:
        newInd += list(np.where(bigDF['Cell type'] == ct)[0])

    bigDF = bigDF.iloc[newInd]
    normalizedData = normalizedData.iloc[newInd]

    # draw two heatmaps, separately for hyper methylated and hypomethylated markers
    fig = plt.figure()

    for ii, hyp in enumerate(['hyper', 'hypo']):
        markerInd = np.hstack([v for _,v in allMarkers[hyp].items()])
        aspect = markerInd.shape[0] / logtpm.shape[0]

        ax = fig.add_subplot(1,2,ii+1)
        ax.imshow(StandardScaler().fit_transform(normalizedData.iloc[:, markerInd]), aspect=aspect, cmap='bwr', vmin=-5,vmax=5)

        c = 0
        r = 0

        yind = []
        cumulativeSamples = 0
        for ct in allCellTypes:
            gg = len(groups[ct])
            c += gg
            if c < 77:
                ax.axhline(c-0.5, color='k', alpha=0.7)

            r += len(allMarkers[hyp][ct])
            if r < markerInd.shape[0]:
                ax.axvline(r-0.5, color='k', alpha=0.7)

            if gg % 2 == 0:
                # even
                loc = (gg // 2) + 0.5
            else:
                loc = (gg // 2) + 1

            yind.append(cumulativeSamples + loc)
            cumulativeSamples += gg

        ax.set_yticks(yind)
        ax.set_yticklabels(allCellTypes)

        mul = markerInd.shape[0] // len(allCellTypes)

        if mul % 2 == 0:
            ad = (mul // 2) + 0.5
        else:
            ad = (mul // 2) + 1


        ax.set_xticks([i*mul+ad for i in range(len(allCellTypes))])
        ax.set_xticklabels(allCellTypes, rotation=45)
        ax.set_title('%s-methylated' % hyp)

    fig.savefig('oki.png')
    sys.exit(0)
    # put the indices of all markers together
    markerHyperInd = np.hstack([v for _,v in allMarkers['hyper'].items()])
    markerHypoInd = np.hstack([v for _,v in allMarkers['hypo'].items()])
    markerInd = np.hstack((markerHyperInd, markerHypoInd))

    # make sure atlas is robust to leaving out one sample

    # set random seeds for reproducibility
    np.random.seed(104)
    # number of leave one sample out repeats
    N = 20
    seeds = np.random.randint(0, 2**32, N)

    # gather all kinds of statistics from these
    corrects = np.zeros(len(allCellTypes))
    tested = np.zeros(len(allCellTypes))
    nmarkers = np.zeros(N)
    nmarkersCt = np.zeros((N, len(allCellTypes)))
    accRound = np.zeros(N)

    # if True, print results of every iteration
    verbose = True
    # whether to cente/scale the data to 0 mean and unit variance
    zscore = True
    # keep false for now, you can look up Variance Stabilizing Transform if you are really interested
    useVst = False

    for i in range(N):
        print(i)
        # leave out one sample from each cell population
        testSamples = splitAtlas(np.array(bigDF.index), np.array(bigDF['Cell type']), seeds[i])

        # left out samples go to testDF, the rest are called trainDF
        trainDF = bigDF.drop(testSamples, axis=0)
        testDF = bigDF.loc[testSamples]

        # get methylation counts
        if not zscore:
            vstTrain = normalizedData.loc[trainDF.index]
            vstTest = normalizedData.loc[testDF.index]
        else:
            # if z-score, estimated mean and variance in the training data
            ss = StandardScaler()
            vstTrain = ss.fit_transform(normalizedData.loc[trainDF.index])
            vstTest = ss.transform(normalizedData.loc[testDF.index])

            vstTrain = pd.DataFrame(data=vstTrain, index=trainDF.index, columns=normalizedData.columns)
            vstTest = pd.DataFrame(data=vstTest, index=testDF.index, columns=normalizedData.columns)


        # make the dictionary with ct as key, methylation profiles as values
        trainDataPerCt = dict()
        for ct in allCellTypes:
            ii = np.where(trainDF['Cell type'] == ct)[0]

            trainDataPerCt[ct] = vstTrain.iloc[ii]

        # fidn markers and isolate these regions
        allMarkers = findMarkersTstat(trainDataPerCt, minTstat, 100)

        for j,ct in enumerate(allCellTypes):
            nmarkersCt[i,j] = len(allMarkers['hyper'][ct]) + len(allMarkers['hypo'][ct])

        markerHyperInd = np.hstack([v for _,v in allMarkers['hyper'].items()])
        markerHypoInd = np.hstack([v for _,v in allMarkers['hypo'].items()])
        markerInd = np.hstack((markerHyperInd, markerHypoInd))

        # isolate marker data
        vstTrainMarkers = vstTrain.iloc[:, markerInd]
        vstTestMarkers = vstTest.iloc[:, markerInd]

        # generate atlas using vst mean markers
        atlasMeans = np.zeros((len(allCellTypes), markerInd.shape[0]))
        atlasSigmas = np.zeros((len(allCellTypes), markerInd.shape[0]))

        nmarkers[i] = atlasMeans.shape[1]

        for j, ct in enumerate(allCellTypes):
            # calculate mean and variance of each marker in each cell type
            x = trainDataPerCt[ct].iloc[:, markerInd]

            atlasMeans[j] = x.mean()
            atlasSigmas[j] = x.std()

        # calculate all pairwise distances between the atlas means and the left out samples
        D = cdist(atlasMeans, vstTestMarkers, metric='cosine')
        # test whether each left out sample is closer to the samples of its correct cell type than other cell types
        correctRound = 0
        for j in range(testDF.shape[0]):
            d = D[:,j]
            ctTrue = testDF.iloc[j]['Cell type']
            ctAtlasInd = allCellTypes.index(ctTrue)
            tested[ctAtlasInd] += 1

            minInd = np.argmin(d)

            if verbose:
                print('%d) true: %s, predicted: %s' % (j, ctTrue, allCellTypes[minInd]))
            if ctTrue == allCellTypes[minInd]:
                if verbose:
                    print('distance to true: %f' % np.min(d))
                    print('distance to 2nd nearest: %f' % np.sort(d)[1])

                corrects[ctAtlasInd] +=1
                correctRound += 1
            else:
                if verbose:
                    print('distance to nearest: %f' % np.min(d))
                    print('distance to true: %f' % d[ctAtlasInd])


        if i == (N-1):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            x = vstTest.iloc[:, markerHyperInd]
            x = x - x.min()
            x = x / x.max()
            ax.imshow(x, aspect=30., cmap='Blues')
            c = 0
            r = 0

            yind = []
            cumulativeSamples = 0
            for ct in allCellTypes:
                gg = len(groups[ct])
                # c += gg
                # if c < 77:
                #     ax.axhline(c-0.5, color='k', alpha=0.7)

                r += len(allMarkers['hyper'][ct])
                if r < markerInd.shape[0]:
                    ax.axvline(r-0.5, color='k', alpha=0.7)

                if gg % 2 == 0:
                    # even
                    loc = (gg // 2) + 0.5
                else:
                    loc = (gg // 2) + 1

                yind.append(cumulativeSamples + loc)
                cumulativeSamples += gg
            ax.set_yticks(np.arange(testDF.shape[0]))
            ax.set_yticklabels(testDF['Cell type'])

            cumulative = 0
            xtick = []

            for k, ct in enumerate(allCellTypes):
                m = cumulative + (len(allMarkers['hyper'][ct]) // 2)
                xtick.append(m)
                cumulative += len(allMarkers['hyper'][ct])

            ax.set_xticks(xtick)
            ax.set_xticklabels(allCellTypes, rotation=45)

        accRound[i] = correctRound / testDF.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)

    ax.hist(nmarkers/len(allCellTypes), edgecolor='k')
    ax.set_xlabel('markers found')


    ax = fig.add_subplot(2,2,2)
    ax.scatter(celltypePresence, corrects/tested)
    ax.set_xlabel('#samples in whole atlas')
    ax.set_ylabel('LOO accuracy for ct')

    ax = fig.add_subplot(2,2,3)
    ax.scatter(nmarkers/len(allCellTypes), accRound)
    ax.set_xlabel('#markers found')
    ax.set_ylabel('LOO accuracy for round')

    # ax = fig.add_subplot(2,2,4)
    # ax.scatter(nmarkers/len(allCellTypes), accRound)
    # ax.set_xlabel('#markers found')
    # ax.set_ylabel('LOO accuracy for round')

    # count how often each cpg island is a marker, takes a while...
    N = 100
    np.random.seed(42)
    seeds = np.random.randint(0, 2**32, N)

    nmark = np.zeros((N, len(allCellTypes), normalizedData.shape[1]), int)

    for i in range(N):
        print('%d/%d' % (i, N))
        testSamples = splitAtlas(np.array(bigDF.index), np.array(bigDF['Cell type']), seeds[i])

        trainDF = bigDF.drop(testSamples, axis=0)
        testDF = bigDF.loc[testSamples]

        vstTrain = normalizedData.loc[trainDF.index]
        vstTest = normalizedData.loc[testDF.index]

        trainDataPerCt = dict()
        for ct in allCellTypes:
            ii = np.where(trainDF['Cell type'] == ct)[0]

            trainDataPerCt[ct] = vstTrain.iloc[ii]

        # fidn markers and isolate these regions
        allMarkers = findMarkersTstat(trainDataPerCt, minTstat)

        for j,ct in enumerate(allCellTypes):
            for k in allMarkers['hyper'][ct]:
                nmark[i, j, k] += 1

            for k in allMarkers['hypo'][ct]:
                nmark[i, j, k] += 1

    # result of nmark, #markers selected 100/100 times
    # settings: zscore, logtpm
    # [13,  3,  2, 13, 42, 44, 48, 52, 59, 53, 25]
    ##########

    # generate atlas and fake mixtures using mean markers
    atlasMeans = np.zeros((len(allCellTypes), markerInd.shape[0]))
    atlasSigmas = np.zeros((len(allCellTypes), markerInd.shape[0]))

    for i, ct in enumerate(allCellTypes):
        x = vstPerCt[ct].iloc[:, markerInd]

        atlasMeans[i] = x.mean()
        atlasSigmas[i] = x.std()

    # final atlas, each row a cell type, each column a marker, contains the mean of each marker in each ct
    atlasDF = pd.DataFrame(atlasMeans, index=allCellTypes, columns=normalizedData.columns[markerInd])

    # save atlas for the future
    if useVst:
        with open('tmp/atlas_new_clusters_vst_%f_tstat.pkl' % minTstat, 'wb') as f:
            pickle.dump({'atlas': atlasDF, 'preprocessing': vstMapper}, f)

    else:
        with open('tmp/atlas_new_clusters_tpm_%f_tstat.pkl' % minTstat, 'wb') as f:
            pickle.dump({'atlas': atlasDF}, f)



    np.random.seed(1)
    # draw artificial mixtures based on the mean and variance of each marker in each cell type
    # this is the easiest possible simulation
    NN = 20

    profiles = np.zeros((NN, markerInd.shape[0]))
    fractions = np.zeros((NN, len(allCellTypes)))

    for i in range(NN):
        # dirichlet distribution can be used to generate in this case 8 probabilities that add to 100%
        # these are the simulated proportions
        ff = dirichlet.rvs(np.ones(len(allCellTypes)))

        # for each marker draw a random number in each cell type
        # (note that here we are making the huge assumption that all markers are independent)
        rr = np.random.normal(atlasMeans, atlasSigmas)

        # each marker value is multiplied by the corresponding cell type proportion and added together
        profiles[i] = ff.dot(rr)
        fractions[i] = ff

    # again save the data
    if useVst:
        with open('tmp/example0_clusters_vst.pkl', 'wb') as f:
            pickle.dump({'atlas': atlasMeans, 'profile': profiles, 'trueprop': fractions, 'ctdraws': rr}, f)
    else:
        with open('tmp/example0_clusters_tpm.pkl', 'wb') as f:
            pickle.dump({'atlas': atlasMeans, 'profile': profiles, 'trueprop': fractions, 'ctdraws': rr}, f)

    # init matlab
    eng = matlab.engine.start_matlab()
    # set working directory to where the code lives
    eng.cd(r'/home/stavros/Desktop/code/deconvolutionsimulations/src/matlab/', nargout=0)

    ########################################################################################################
    # estimate proportions

    # estimated fractions
    ef = np.zeros(fractions.shape)
    # linear and non-linear correlation between true and estimated fractions
    rhoSpearman = np.zeros(ef.shape[0])
    rhoPearson = np.zeros(ef.shape[0])

    # convert data frame to np array, because matlab doesn't like pandas
    atlas = np.array(atlasDF)

    for i in range(profiles.shape[0]):
        # call the deconvolution function from matlab one mixture at a time
        # this takes > 1 min per profile, so this will take a couple of hours
        matlabres = eng.deconvolve(profiles[i], atlas, 1.0, 0.1, nargout=1)
        # save result
        ef[i] = np.array(matlabres).reshape(-1,)

        # calculate correlations
        rhoPearson[i] = pearsonr(ef[i], fractions[i])[0]
        rhoSpearman[i] = spearmanr(ef[i], fractions[i])[0]

        print('Iteration %d:, r = %.3f' % (i, rhoPearson[i]))

    # do the same, now not for mixtures, but individual samples from 1 ct
    ef2 = np.zeros((atlasDF.shape[0], atlasDF.shape[0]))
    for i in range(rr.shape[0]):
        matlabres = eng.deconvolve(rr[i], atlas, 1.0, 0.1, nargout=1)

        ef2[i] = np.array(matlabres).reshape(-1,)

    for i in range(ef2.shape[0]):
        # this number should be close to 1 (100% proportions of a single ct)
        print(ef2[i,i])

    # plot distribution of correlations
    xx = np.linspace(-1,1,2000)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = gaussian_kde(rhoPearson)
    ax.plot(xx, d(xx), label='pearson')
    d = gaussian_kde(rhoSpearman)
    ax.plot(xx, d(xx), label='spearman')

    ax.legend()
    # plt.show()


    # # same but odds are not uniform
    # np.random.seed(1)
    # # draw 100 artificial mixtures
    # NN = 10
    #
    # odds = {'B-cell': 1.,
    #  'monocytes': 2.,
    #  'granulocytes': 3.,
    #  'T-cell': 2.,
    #  'colon': 1.,
    #  'melanoma': 0.8,
    #  'neuron': 0.8,
    #  'oligodendrocyte': 0.8}
    #
    # oddsVector = [odds[k] for k in allCellTypes]
    #
    # profiles = np.zeros((NN, markerInd.shape[0]))
    # fractions = np.zeros((NN, len(allCellTypes)))
    #
    # for i in range(NN):
    #
    #     ff = dirichlet.rvs(oddsVector)
    #
    #     rr = np.random.normal(atlasMeans, atlasSigmas)
    #
    #     profiles[i] = ff.dot(rr)
    #     fractions[i] = ff
    #
    # if useVst:
    #     with open('tmp/example0.5_clusters_vst.pkl', 'wb') as f:
    #         pickle.dump({'atlas': atlasMeans, 'profile': profiles, 'trueprop': fractions, 'ctdraws': rr}, f)
    # else:
    #     with open('tmp/example0.5_clusters_tpm.pkl', 'wb') as f:
    #         pickle.dump({'atlas': atlasMeans, 'profile': profiles, 'trueprop': fractions, 'ctdraws': rr}, f)


    ###################################
    print('Leaving samples out...')
    np.random.seed(123456)
    N = 50
    seeds = np.random.randint(0, 2**32, N)

    odds = {'B-cell': 1.,
     'monocytes': 2.,
     'granulocytes': 3.,
     'T-cell': 2.,
     'colon': 1.,
     'melanoma': 0.8,
     'neuron': 0.8,
     'oligodendrocyte': 0.8}


    atlases = []
    mixtures = {0: [], 0.005: [], 0.01: [], 0.05: [], 0.1: [], 0.5: [], 1.0: []}

    noiseInds = sorted(list(mixtures.keys()))[1:]

    bigDF2 = bigDF
    allCellTypes2 = allCellTypes

    trueFracs = np.zeros((N, len(allCellTypes2)))
    estimatedFracs = np.zeros((N, len(mixtures), len(allCellTypes2)))

    pearsonRhos = np.zeros((N, len(noiseInds)+1))

    for i in range(N):
        print(i)
        testSamples = splitAtlas(np.array(bigDF2.index), np.array(bigDF2['Cell type']), seeds[i])

        trainDF = bigDF2.drop(testSamples, axis=0)
        testDF = bigDF2.loc[testSamples]

        ss = StandardScaler()
        vstTrain = ss.fit_transform(normalizedData.loc[trainDF.index])
        vstTest = ss.transform(normalizedData.loc[testDF.index])

        vstTrain = pd.DataFrame(data=vstTrain, index=trainDF.index, columns=normalizedData.columns)
        vstTest = pd.DataFrame(data=vstTest, index=testDF.index, columns=normalizedData.columns)


        trainDataPerCt = dict()
        for ct in allCellTypes2:
            ii = np.where(trainDF['Cell type'] == ct)[0]

            trainDataPerCt[ct] = vstTrain.iloc[ii]

        # fidn markers and isolate these regions
        allMarkers = findMarkersTstat(trainDataPerCt, minTstat, 100)

        markerHyperInd = np.hstack([v for _,v in allMarkers['hyper'].items()])
        markerHypoInd = np.hstack([v for _,v in allMarkers['hypo'].items()])
        markerInd = np.hstack((markerHyperInd, markerHypoInd))
        print(markerInd.shape)

        # markerInd = np.hstack([v for _,v in allMarkers.items()])
        vstTrainMarkers = vstTrain.iloc[:, markerInd]
        vstTestMarkers = vstTest.iloc[:, markerInd]

        # generate atlas using vst mean markers
        atlasMeans = np.zeros((len(allCellTypes2), markerInd.shape[0]))
        atlasSigmas = np.zeros((len(allCellTypes2), markerInd.shape[0]))


        for j, ct in enumerate(allCellTypes2):
            x = trainDataPerCt[ct].iloc[:, markerInd]

            atlasMeans[j] = x.mean()
            atlasSigmas[j] = x.std()

        atlases.append(atlasMeans)

        # D = cdist(atlasMeans, vstTestMarkers, metric='cosine')
        # for j in range(testDF.shape[0]):
        #     d = D[:,j]
        #     ctTrue = testDF.iloc[j]['Cell type']
        #     ctAtlasInd = allCellTypes2.index(ctTrue)
        #
        #     minInd = np.argmin(d)
        #
        #     print('%d) true: %s, predicted: %s' % (j, ctTrue, allCellTypes2[minInd]))
        #     if ctTrue == allCellTypes2[minInd]:
        #         print('distance to true: %f' % np.min(d))
        #         print('distance to 2nd nearest: %f' % np.sort(d)[1])
        #     else:
        #         print('distance to nearest: %f' % np.min(d))
        #         print('distance to true: %f' % d[ctAtlasInd])

        # make sure true fractions are always lined-up in terms of the alphabetical order of cell types
        internalFractions = dirichlet.rvs(np.array(testDF['Cell type'].map(odds))).reshape(-1,)
        for j in range(testDF.shape[0]):
            jind = allCellTypes2.index(testDF['Cell type'].iloc[j])
            trueFracs[i, jind] = internalFractions[j]


        profile = internalFractions.dot(vstTestMarkers)
        mixtures[0].append(profile)

        matlabres = eng.deconvolve(profile, atlasMeans, 1.0, 0.1, nargout=1)
        # save result
        estimatedFracs[i,0] = np.array(matlabres).reshape(-1,)
        pearsonRhos[i,0] = pearsonr(trueFracs[i], estimatedFracs[i,0])[0]
        print(pearsonRhos[i,0])

        for j, noise in enumerate(noiseInds, 1):
            mixtures[noise].append(profile + np.random.normal(0, noise, markerInd.shape[0]))
            matlabres = eng.deconvolve(mixtures[noise][-1], atlasMeans, 1.0, 0.1, nargout=1)

            estimatedFracs[i,j] = np.array(matlabres).reshape(-1,)
            pearsonRhos[i,j] = pearsonr(trueFracs[i], estimatedFracs[i,j])[0]
            print(pearsonRhos[i,j])



    df = pd.DataFrame({'rho': pearsonRhos.flatten(), 'noise': 20*sorted(list(mixtures.keys()))})
    sns.boxplot(data=df,x='noise',y='rho')
    sns.swarmplot(data=df,x='noise',y='rho', color='k')

    plt.show()
    sys.exit(0)

    ########################
    import matlab.engine
    import pickle
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr, gaussian_kde
    import numpy as np

    uniform = False

    if uniform:
        # with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example0.pkl', 'rb') as f:
        with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example0_clusters_tpm.pkl', 'rb') as f:
                    dataDict = pickle.load(f)
    else:
        # with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example0_5.pkl', 'rb') as f:
        with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example0.5_clusters_tpm.pkl', 'rb') as f:
            dataDict = pickle.load(f)

    # with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example1.pkl', 'rb') as f:
    with open('/home/stavros/emc/users/smakrodimitris/edit/src/tmp/example1_clusters_tpm.pkl', 'rb') as f:
        dataDict1 = pickle.load(f)

    # init matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/stavros/Desktop/code/deconvolutionsimulations/src/matlab/', nargout=0)

    ########################################################################################################
    # example 0
    ef = np.zeros(dataDict['trueprop'].shape)
    rhoSpearman = np.zeros(ef.shape[0])
    rhoPearson = np.zeros(ef.shape[0])


    for i in range(dataDict['profile'].shape[0]):
        matlabres = eng.deconvolve(dataDict['profile'][i], dataDict['atlas'], 1.0, 0.1, nargout=1)

        ef[i] = np.array(matlabres).reshape(-1,)

        rhoPearson[i] = pearsonr(ef[i], dataDict['trueprop'][i])[0]
        rhoSpearman[i] = spearmanr(ef[i], dataDict['trueprop'][i])[0]

        print('Iteration %d:, r = %.3f' % (i, rhoPearson[i]))

    ef2 = np.zeros((dataDict['atlas'].shape[0], dataDict['atlas'].shape[0]))
    for i in range(dataDict['ctdraws'].shape[0]):
        matlabres = eng.deconvolve(dataDict['ctdraws'][i], dataDict['atlas'], 1.0, 0.1, nargout=1)

        ef2[i] = np.array(matlabres).reshape(-1,)

    for i in range(ef2.shape[0]):
        print(ef2[i,i])


    xx = np.linspace(-1,1,2000)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = gaussian_kde(rhoPearson)
    ax.plot(xx, d(xx), label='pearson')
    d = gaussian_kde(rhoSpearman)
    ax.plot(xx, d(xx), label='spearman')

    ax.legend()
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(tf,ef)

    ########################################################################################################
    # example 1
    N = dataDict1['trueprop'].shape[0]
    noises = sorted(list(dataDict1['profile'].keys()))

    ef = np.zeros((N, len(noises), dataDict1['trueprop'].shape[1]))
    rhoSpearman = np.zeros(ef.shape[:2])
    rhoPearson = np.zeros(ef.shape[:2])

    for j, n in enumerate(noises):
        for i in range(N):
            matlabres = eng.deconvolve(dataDict1['profile'][n][i], dataDict1['atlas'][i], 1.0, 0.1, nargout=1)

            ef[i, j] = np.array(matlabres).reshape(-1,)

            rhoPearson[i, j] = pearsonr(ef[i, j], dataDict1['trueprop'][i])[0]
            rhoSpearman[i, j] = spearmanr(ef[i, j], dataDict1['trueprop'][i])[0]

            print('Iteration %d, noise %f:' % (i, n))
            print('rho = %.3f' % rhoPearson[i, j])



    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    from scipy.stats import gaussian_kde

    dens = gaussian_kde(rhoPearson[:,0])
    xx = np.linspace(0.7, 1.0, 50)

    ax.plot(xx, dens(xx), color='k')
    ax.scatter(rhoPearson[:,0], np.zeros(rhoPearson.shape[0]), color='k', alpha=0.5)

    ax.set_title('noise sigma = 0.0' )

    ax.set_xlabel('correlation')

    ax = fig.add_subplot(1,2,2)

    dens = gaussian_kde(rhoPearson[:,1])
    xx = np.linspace(0.7, 1.0, 50)

    ax.plot(xx, dens(xx), color='k')
    ax.scatter(rhoPearson[:,1], np.zeros(rhoPearson.shape[0]), color='k', alpha=0.5)

    ax.set_title('noise sigma = 0.1' )
    ax.set_xlabel('correlation')




    ########################################################################################################
    # hyperparameter search
    Cs = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.]
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    performances = np.zeros((len(Cs), len(epsilons), ef.shape[0]))
    for i, CC in enumerate(Cs):
        for j, ee in enumerate(epsilons):

            for k in range(dataDict['profile'].shape[0]):
                matlabres = eng.deconvolve(dataDict['profile'][k], dataDict['atlas'], CC, ee, nargout=1)

                tmpEf = np.array(matlabres).reshape(-1,)

                performances[i,j,k] = pearsonr(tmpEf, dataDict['trueprop'][k])[0]

    # C = 1 and eps = 0.1 are pretty good, eps = 0.1 clearly the best choice
    # C less important if eps is at 0.1

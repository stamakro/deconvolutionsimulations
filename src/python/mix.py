import numpy as np
import os
from convertCounts2genes import convertCounts2clusteredRegions
import sys
import pickle
import pandas as pd
import json

class MedseqRun():
    def __init__(self, filename: str):
        # read an rcsite file and return a dictionary with site id and read count
        # only for sites with reads
        self.counts = dict()
        self.totalReads = 0
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                c = int(line.split('\t')[0])

                if c > 0:
                    self.counts[i] = c
                    self.totalReads += c



    def sample(self, target: int, seed=None):
        # target: number of reads to keep
        # seed: random seed (optional)
        # downsample a set of reads to user-specified target value, by removing reads

        if target > self.totalReads:
            print('not enough reads! Asked for %d, but sample has %d.' % (target, self.totalReads))
            raise ValueError

        if seed is not None:
            np.random.seed(seed)

        # all regions that have non-zero count
        locations = sorted(self.counts.keys())

        regions = np.zeros(self.totalReads, int)
        current = 0
        for l in locations:
            regions[current:current+self.counts[l]] = l

            current += self.counts[l]


        # result is stored here
        sampledCounts = dict()
        samples = np.random.choice(regions, size=target, replace=False)

        sampledRegions, counts = np.unique(samples, return_counts=True)

        for rr, cc in zip(sampledRegions, counts):
            sampledCounts[rr] = cc

        return sampledCounts


def writeToFile(filename, countsDict):
    with open(filename, 'w') as fw:
        for i in range(16596934):
            try:
                c = countsDict[i]
            except KeyError:
                c = 0
            line = '%d\t%d\t%d\t%d\n' % (c, i, c, 0)
            fw.write(line)



def mix(samples: list, proportions: list, totalcoverage: int, seed: int):
    # sampleFiles: a list of medseq objects to sample from
    # proportions: list of the corresponding proportions
    # total # methylated reads to generate in the mixture
    # returns a dictionary of counts

    assert len(samples) == len(proportions), 'Length mismatch samples and proportions'
    proportions = np.array(proportions)
    assert np.sum(proportions) == 1.0, 'Provided mixture proportions do not add to 1'

    neededReads = (totalcoverage * proportions).astype(int)
    # print(neededReads)
    # assert np.min(neededReads) > 0, 'Specified proportions of at least one sample too low'

    mixture = None
    for i, (mo,r) in enumerate(zip(samples, neededReads)):
        # if no reads required for one sample do nothing
        if r > 0:
            # otherwise subsample each sample and add the reads
            # to the pile
            sampledCounts = mo.sample(r, seed)

            if mixture is None:
                mixture = sampledCounts
            else:
                # add sampled read counts to existing list of read counts
                for k,v in sampledCounts.items():
                    if k in mixture:
                        mixture[k] += v
                    else:
                        mixture[k] = v


    actualProportions = neededReads / np.sum(neededReads)

    return (mixture, list(actualProportions))


def convert2df(countsDict: dict):
    cc = np.zeros(16596934, int)
    for cpg, count in countsDict.items():
        cc[cpg] = count

    return cc




if __name__ == '__main__':
    # neuronName = '../data/atlas/neuron_2_CpG_M_rcSite.txt'
    # healthyName = '../../hbds/batch2/L9508_rcSite.txt'
    #
    # neuron = MedseqRun('../data/atlas/neuron_2_CpG_M_rcSite.txt')
    # healthy = MedseqRun('../../hbds/batch2/L9508_rcSite.txt')
    #
    # mixture = mix([neuron, healthy], [0.1, 0.9], 5000000, 42)

    inputPathsFile = sys.argv[1]
    with open(inputPathsFile) as f:
        inputPaths = json.load(f)


    allSampleNames = []
    allSampleCodeNames = []
    for code, path in inputPaths.items():

        tmpFiles = []
        tmpNames = []
        for k in sorted(os.listdir(path)):
            tmpFiles.append(path + k)
            # keep first part before all '_'
            # then remove all '-'
            cc = k.split('_')[0]
            cc = ''.join(cc.split('-'))
            tmpNames.append(code + cc)


        allSampleNames.append(tmpFiles)
        allSampleCodeNames.append(tmpNames)


    Ntissues = len(allSampleNames)

    print('reading samples...')
    allSamples = [[MedseqRun(f) for f in ss] for ss in allSampleNames]

    # allNeuronSamples = [MedseqRun(f) for f in allNeuronSampleNames]
    # print('reading healthy samples...')
    # allHBDSamples = [MedseqRun(f) for f in allHBDSampleNames]
    print('starting to mix')

    savePath = sys.argv[2]

    # nr of mixtures

    Nsamples = 30

    # nr of repeats/seeds to be used for mixing
    seeds = [42, 10000, 192071]

    if Ntissues != 2:
        raise NotImplementedError

    # for two tissues, assume background is the second
    # fractions = [1.0, 0.5, 0.3, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0]
    fractions = np.array([[1.0, 0.0], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9], [0.075, 0.925], [0.01, 0.99], [0.005, 0.995], [0.001, 0.999], [0.0, 1.0]])

    depths = [5000000, 10000000]


    # 11 fractions * 30 mixtures * 3 repeats * 2 depths = 1980 fake medseq samples
    np.random.seed(104)
    counter = 0

    fakeClusters = np.zeros(((Nsamples * len(fractions) * len(depths) * len(seeds)), 392443), int)
    mixedSampleNames = []

    for i in range(Nsamples):

        indices = [np.random.choice(len(ss)) for ss in allSamples]

        for dp in depths:
            for nfr in fractions:
                for seed in seeds:
                    print(counter, flush=True)
                    try:
                        mixedCounts, proportions = mix([k[ii] for (k,ii) in zip(allSamples, indices)], nfr, dp, seed)
                        #sn = '%s-%.4f-%s-%.4f-%d-%d' % (codenamesHBD[hsi], proportions[1], codenamesNeurons[nsi], proportions[0], dp, seed)
                        sn = ''
                        for tc, (nn, ii) in enumerate(zip(allSampleCodeNames, indices)):
                            sn += '%s-%.4f-' % (nn[ii], proportions[tc])
                        sn += '%d-%d' % (dp, seed)

                        print(sn)
                        clusterProfile = convert2df(mixedCounts)
                        fakeClusters[counter] = convertCounts2clusteredRegions(clusterProfile, 0)

                    except ValueError:
                        sn = 'invalid-sim'

                    mixedSampleNames.append(sn)
                    # writeToFile(outfile, mixedCounts)
                    counter += 1


    with open('../resources/site2cluster.pkl', 'rb') as f:
        mapping = pickle.load(f)

    geneNames = ['' for _ in range(len(mapping))]

    for ii, key in enumerate(sorted(mapping)):
        cpgInds = mapping[key]
        geneNames[ii] = key

    df = pd.DataFrame(fakeClusters, index=mixedSampleNames, columns=geneNames)
    # df.to_csv(savePath + 'sim1.csv')

    # save in smaller files to be read in parallel more quickly
    nfiles = 1 + (df.shape[0] // 100)
    for i in range(nfiles):
        df.iloc[i*100:(i+1)*100].to_csv(savePath + 'sim1_part' + str(i) + '.csv')

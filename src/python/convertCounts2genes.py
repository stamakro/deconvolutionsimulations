from pathlib import Path
import sys
import pandas as pd
from copy import deepcopy
import numpy as np
import os
import pickle
import warnings

def createPath(inputPath):
	srcPath = Path(__file__).parent
	return (srcPath / inputPath).resolve()


def generateSite2tss():
	genes = pd.read_table(createPath('../resources/2kbTSS.tsv'))

	sites = np.array(genes['SiteIds'])
	siteCounts = np.array(genes['SiteCount'])

	geneNames =  list(genes['Chromosome']+':'+genes['Start'].astype(str) + '-' + genes['End'].astype(str))
	uniqueGeneNames = []
	geneSet = set()
	for g in geneNames:
		if g not in geneSet:
			uniqueGeneNames.append(g)
			geneSet.add(g)

	#uniqueGenes = np.unique(geneNames)

	g2i = dict()
	for i, g in enumerate(uniqueGenes):
		g2i[g] = i

	mapping = dict()
	for i, (s, sc, g) in enumerate(zip(sites, siteCounts, geneNames)):

		if s != 'EMPTY;':
			ss = s.split(';')
			start = int(ss[0])
			try:
				end = int(ss[1])
			except ValueError:
				assert sc == 1 or sc == 0
				end = start
		else:
			start = -1
			end = -1


		if g in mapping:
			assert mapping[g][0] == start
			assert mapping[g][1] == end
		else:
			mapping[g] = [start, end, g2i[g]]


	with open(createPath('../resources/site2tss.pkl'), 'wb') as f:
		pickle.dump(mapping, f)



def generateSite2mhb():
	genes = pd.read_table(createPath('../resources/mhb-medseq-overlap.txt'), header=None, usecols=[4,5,6,7], index_col=0)

	geneNames =  list(genes[5]+':'+genes[6].astype(str) + '-' + genes[7].astype(str))
	genes[1] = genes[5]+':'+genes[6].astype(str) + '-' + genes[7].astype(str)
	genes.drop([5,6,7], axis=1, inplace=True)
	uniqueGenes = np.unique(geneNames)

	g2i = dict()
	for i, g in enumerate(uniqueGenes):
		g2i[g] = i

	mapping = dict()

	for i, gg in zip(genes.index, genes[1]):
		if gg not in mapping:
			mapping[gg] = [g2i[gg], i]
		else:
			mapping[gg].append(i)


	with open('resources/site2mhb.pkl', 'wb') as f:
		pickle.dump(mapping, f)

def generateSite2cpgIsland():
	mapping = dict()

	with open(createPath('../resources/cpg_islands_intersect_lpnp1_sites.txt')) as f:
		for line in f:
			fields = line.split('\t')

			island = fields[0] + ':' + fields[1] + '-' + fields[2]

			rcSite = int(fields[-1])

			if rcSite == -1:
				continue

			if island in mapping:
				mapping[island].append(rcSite)
			else:
				mapping[island] = [rcSite]

	with open(createPath('../resources/site2island.pkl'), 'wb') as f:
		pickle.dump(mapping, f)



def convertCounts2tss(rawCounts, ind=None):

	if ind is None:
		ind = [0, 1, 2, 3]

	try:
		with open(createPath('../resources/site2tss.pkl'), 'rb') as f:
			mapping = pickle.load(f)
	except FileNotFoundError:
			generateSite2tss()
			with open(createPath('../resources/site2tss.pkl'), 'rb') as f:
				mapping = pickle.load(f)

	genes = pd.read_table(createPath('../resources/2kbTSS.tsv'))

	transformedCounts = np.zeros((len(mapping), 4))

	geneNames = ['' for _ in range(len(mapping))]

	for key in mapping:
		start, end, index = mapping[key]


		geneNames[index] = key
		if start > -1 and end > -1:
			tmp = rawCounts[start:(end+1)]
			transformedCounts[index] = [np.sum(tmp), np.max(tmp), np.min(tmp), np.sum(tmp > 0)]
		else:
			transformedCounts[index] = [np.nan, np.nan, np.nan, np.nan]


	result = pd.DataFrame(data=transformedCounts, index=geneNames, columns=['sum', 'max', 'min', 'nsitesActive'])
	result = result.iloc[:, ind]

	return result


def convertCounts2mhb(rawCounts, ind=None):

	if ind is None:
		ind = [0, 1, 2, 3, 4, 5]

	try:
		with open(createPath('../resources/site2mhb.pkl'), 'rb') as f:
			mapping = pickle.load(f)
	except FileNotFoundError:
			generateSite2mhb()
			with open(createPath('resources/site2mhb.pkl'), 'rb') as f:
				mapping = pickle.load(f)

	transformedCounts = np.zeros((len(mapping), 6))

	geneNames = ['' for _ in range(len(mapping))]

	keys = sorted(list(mapping.keys()))

	for key in keys:
		loc = mapping[key][0]
		index = mapping[key][1:]

		geneNames[loc] = key

		tmp = rawCounts[index]
		transformedCounts[loc] = [np.sum(tmp), np.max(tmp), np.min(tmp), np.sum(tmp > 0), np.mean(tmp), np.std(tmp, ddof=1)]

	result = pd.DataFrame(data=transformedCounts, index=geneNames, columns=['sum', 'max', 'min', 'nsitesActive', 'mean', 'std'])
	result = result.iloc[:, ind]

	return result



def convertCounts2CpGisland(rawCounts, ind=None):

	if ind is None:
		ind = [0, 1, 2, 3]

	try:
		with open(createPath('../resources/site2island.pkl'), 'rb') as f:
			mapping = pickle.load(f)
	except FileNotFoundError:
			generateSite2cpgIsland()
			with open(createPath('../resources/site2island.pkl'), 'rb') as f:
				mapping = pickle.load(f)

	genes = pd.read_table(createPath('../resources/cpg_islands_grch38_chromosomes_sorted.bed'), names=["chrom", "start", "end", "name"])

	genes["ID"] = genes["chrom"] + ":" + genes["start"].astype(str) + '-' + genes["end"].astype(str)
	genes.drop('name', axis=1, inplace=True)
	genes.set_index('ID', inplace=True)


	transformedCounts = np.zeros((len(mapping), 4))

	geneNames = ['' for _ in range(len(mapping))]

	for ii, key in enumerate(sorted(mapping)):
		cpgInds = mapping[key]


		geneNames[ii] = key

		tmp = rawCounts[cpgInds]
		transformedCounts[ii] = [np.sum(tmp), np.max(tmp), np.min(tmp), np.sum(tmp > 0)]

	result = pd.DataFrame(data=transformedCounts, index=geneNames, columns=['sum', 'max', 'min', 'nsitesActive'])
	result = result.iloc[:, ind]

	return result


def convertCounts2clusteredRegions(rawCounts, ind=None):

	if ind is None:
		ind = [0, 1, 2, 3]

	try:
		with open(createPath('/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/src/python/resources/site2cluster.pkl'), 'rb') as f:
			mapping = pickle.load(f)
	except FileNotFoundError:
			# TODO : not implemented in codebase
			generateSite2cluster()
			with open(createPath('resources/site2cluster.pkl'), 'rb') as f:
				mapping = pickle.load(f)

	transformedCounts = np.zeros((len(mapping), 4))

	geneNames = ['' for _ in range(len(mapping))]

	for ii, key in enumerate(sorted(mapping)):
		cpgInds = mapping[key]
		geneNames[ii] = key

		tmp = rawCounts[cpgInds]
		transformedCounts[ii] = [np.sum(tmp), np.max(tmp), np.min(tmp), np.sum(tmp > 0)]

	result = pd.DataFrame(data=transformedCounts, index=geneNames, columns=['sum', 'max', 'min', 'nsitesActive'])
	result = result.iloc[:, ind]

	return result



def convertSite2all(rcSiteFile, rcsite=False, cpgi=False, tss=False, mhb=False, clus=False, ind=None, return_library_size=True):
	assert (rcsite or cpgi or tss or mhb or clusters), 'At least one of rcsite, cpgi, tss, mhb must be True'

	rawCounts = np.array(pd.read_table(rcSiteFile, header=None, usecols=[0])).reshape(-1,)
	returnDict = dict()

	if rcsite:
		returnDict['rcsite'] = rawCounts

	if return_library_size:
		returnDict['libsize'] = np.sum(rawCounts)

	if cpgi:
		returnDict['cpgi'] = convertCounts2CpGisland(rawCounts, ind)

	if tss:
		returnDict['tss'] = convertCounts2tss(rawCounts, ind)

	if mhb:
		returnDict['mhb'] = convertCounts2mhb(rawCounts, ind)

	if clus:
		returnDict['clus'] = convertCounts2clusteredRegions(rawCounts, ind)


	return returnDict


# these are test functions to make sure the 2kbTSS code works; they use data that are not available
# TODO: delete for public release?
def verify():
	warnings.warn("deprecated", DeprecationWarning)
	prefix = '../cf-DNA-Teoman-Stavros-data/Reads-per-LpnPI-CpG/'
	files = os.listdir(prefix)

	for i, file in enumerate(files):
		print('%d/%d' % (i, len(files)), flush=True)
		nn = file.split('_')[0]

		example = pd.read_table('../cfDNA-Teoman/2kbTSS/' + nn + '_1kbTSS1kb_Lpnp1CpG_ReadCount.txt')

		result = convert(prefix + file)

		print('Converted. Now checking...')
		assert not (np.array(result['Score']) != np.array(example['Score']) ).any()
		assert not (np.array(result['Max']) != np.array(example['Max']) ).any()
		assert not (np.array(result['Min']) != np.array(example['Min']) ).any()
		assert not (np.array(result['SiteCountActive']) != np.array(example['SiteCountActive']) ).any()



def convertHealthy():
	warnings.warn("deprecated", DeprecationWarning)
	prefix = '../Healthy-cfDNA/'
	files = os.listdir(prefix)

	for i, file in enumerate(files):
		print('%d/%d' % (i, len(files)))
		nn = file.split('_')[0]

		result = convert(prefix + file)

		result.to_csv(prefix + '2kbTSS_' + file.split('_')[0] , sep='\t', index=False)

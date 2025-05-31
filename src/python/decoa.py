import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import json
import sys

def renameSample(names: list) -> dict:
    '''
    fix the problem where some samples are number 1-N vs neuron_1
    names:  list, a list of all sample names
    RETURNS a dictionary which can be used to fix the names by DataFrame.rename
    '''
    renameDict = dict()
    for name in names:
        if 'neuron' in name:
            number = int(name.split('_')[1])
            renameDict[name] = str(number) + '-' + 'N'
        elif 'oligodendrocyte' in name:
            number = int(name.split('_')[1])
            renameDict[name] = str(number) + '-' + 'O'
    return renameDict

def geneLength(regionName: str) -> int:
    '''
    Calculate the length of a genomic region from a string like chr1:1000-2000
    '''
    try:
        start_end = regionName.split(':')[1]
        start, end = start_end.split('-')
        return int(end) - int(start)
    except:
        return 1  # fallback if parsing fails

def findMarkersTstat(vstPerCt: dict, minTstat: float, maxMarkers: int=500):
    '''
    Find marker regions per cell type using t-statistics
    vstPerCt: dict of DataFrames, keys=cell types, values=normalized data for that cell type
    minTstat: minimum absolute t-statistic to consider a marker
    maxMarkers: maximum number of markers per cell type
    Returns dict with keys 'hyper' and 'hypo', each mapping to dicts of ct: list of marker indices
    '''
    allMarkers = {'hyper': {}, 'hypo': {}}
    all_cts = list(vstPerCt.keys())
    n_features = vstPerCt[all_cts[0]].shape[1]

    for ct in all_cts:
        other_cts = [c for c in all_cts if c != ct]
        markers_hyper = []
        markers_hypo = []
        for i in range(n_features):
            group1 = vstPerCt[ct].iloc[:, i]
            group2_vals = []
            for other in other_cts:
                group2_vals.append(vstPerCt[other].iloc[:, i])
            group2 = pd.concat(group2_vals)

            try:
                tstat, pval = ttest_ind(group1, group2, equal_var=False)
            except:
                continue
            if np.isnan(tstat):
                continue
            if tstat > minTstat:
                markers_hyper.append(i)
            elif tstat < -minTstat:
                markers_hypo.append(i)
        # Limit markers to maxMarkers
        allMarkers['hyper'][ct] = markers_hyper[:maxMarkers]
        allMarkers['hypo'][ct] = markers_hypo[:maxMarkers]

    return allMarkers

def splitAtlas(sample_names: np.ndarray, cell_types: np.ndarray, seed: int):
    '''
    Leave-one-out sample selection for atlas building
    sample_names: array of sample names
    cell_types: array of corresponding cell types
    seed: random seed for reproducibility
    Returns list of sample names to leave out (one per cell type)
    '''
    np.random.seed(seed)
    leave_out = []
    unique_cts = np.unique(cell_types)
    for ct in unique_cts:
        indices = np.where(cell_types == ct)[0]
        if len(indices) == 0:
            continue
        chosen = np.random.choice(indices, 1)[0]
        leave_out.append(sample_names[chosen])
    return leave_out

# ----- MAIN SCRIPT -----

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python decoa.py <input_mixtures.json> <output_predictions.json>")
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]
############## change
    sampleInfoPath = 'atlas'
    outpath = 'afroditi'


    # Chromosomes to keep (no sex chromosomes)
    acceptedChromosomes = set(['chr' + str(i) for i in range(1, 23)])

    # Load methylation counts and sample info
    data = pd.read_csv(outputPath + 'counts_aggregated_clusters.csv', index_col=0).T
    sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=';')
    if sampleInfo.shape[1] == 0:
        sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=',')

    # Fix neuron and oligodendrocyte sample names
    rd = renameSample(sampleInfo.index)
    sampleInfo.rename(index=rd, inplace=True)

    # Align sample info and data
    sampleInfo = sampleInfo.loc[data.index]
    assert (sampleInfo.index == data.index).all()

    # Merge into big DataFrame
    bigDF = pd.concat((data, sampleInfo), axis=1, ignore_index=False)

    # Filter low quality samples
    bigDF = bigDF[bigDF['CpG/Total'] > 20.]
    bigDF = bigDF[bigDF['Used reads'] > 3000000]

    # Filter columns to autosomal chromosomes only
    region_cols = bigDF.columns[pd.Series(bigDF.columns).apply(lambda x: x.split(':')[0]).isin(acceptedChromosomes)]
    sampleCountsAutosomal = bigDF[region_cols]

    # Calculate lengths and TPM normalization
    countsAutosomal = np.array(sampleCountsAutosomal, int)
    lengths = np.array(pd.Series(sampleCountsAutosomal.columns).apply(geneLength))
    countsPerBase = countsAutosomal / lengths
    total = np.sum(countsPerBase, axis=1)
    tpm = 1e6 * (countsPerBase.T / total).T
    logtpm = np.log(tpm + 1)
    assert not np.sum(np.isnan(logtpm))

    normalizedData = pd.DataFrame(data=logtpm, index=sampleCountsAutosomal.index, columns=sampleCountsAutosomal.columns)

    # Cell type info
    allCellTypes = sorted(bigDF['Cell type'].unique())

    # Track cell type counts
    cellTypePresenceDraft = bigDF.groupby('Cell type').count().iloc[:, 0].to_dict()
    celltypePresence = np.array([cellTypePresenceDraft[ct] for ct in allCellTypes])

    # Split data by cell type
    vstPerCt = dict()
    for ct in allCellTypes:
        idx = np.where(bigDF['Cell type'] == ct)[0]
        vstPerCt[ct] = normalizedData.iloc[idx]

    # Find markers with t-stat >= 1.5
    minTstat = 1.5
    allMarkers = findMarkersTstat(vstPerCt, minTstat)

    # Reorder samples so same cell types are together
    groups = bigDF.groupby('Cell type').groups
    newInd = []
    for ct in allCellTypes:
        newInd += list(np.where(bigDF['Cell type'] == ct)[0])
    bigDF = bigDF.iloc[newInd]
    normalizedData = normalizedData.iloc[newInd]

    # Combine marker indices
    markerHyperInd = np.hstack([v for _, v in allMarkers['hyper'].items()])
    markerHypoInd = np.hstack([v for _, v in allMarkers['hypo'].items()])
    markerInd = np.hstack((markerHyperInd, markerHypoInd))

    # Build atlasMeans and atlasSigmas (mean and std for each cell type over markers)
    atlasMeans = np.zeros((len(allCellTypes), markerInd.shape[0]))
    atlasSigmas = np.zeros_like(atlasMeans)

    for j, ct in enumerate(allCellTypes):
        x = vstPerCt[ct].iloc[:, markerInd]
        atlasMeans[j] = x.mean()
        atlasSigmas[j] = x.std()

    # Load fake mixtures and deconvolve using cosine distance
    #if len(sys.argv) <2:
     #   print("python decoa.py <path_to_mixes_json>")
      #  sys.exit(1)
    
    mixture_json_path = sys.argv[1]

    with open(mixture_json_path), 'r') as f:
        fake_mixtures_data = json.load(f)

    fake_mixtures = ps.DataFrame(fake_mixtures_data).T
    # Filter fake mixtures to marker columns only
    fake_markers = fake_df.iloc[:, markerInd]

    # Cosine distance between atlas means and fake mixtures
    D = cdist(atlasMeans, fake_markers.values, metric='cosine')

    # Predict cell types for fake mixtures
    predicted_cts = [allCellTypes[np.argmin(D[:, i])] for i in range(D.shape[1])]

    # Print predicted cell types
    for i, sample_name in enumerate(fake_markers.index):
        print(f'Sample {sample_name} predicted as cell type: {predicted_cts[i]}')


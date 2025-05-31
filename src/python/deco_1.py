import sys
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import cdist


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

def geneLength(region: str) -> int:
    # takes a region, e.g. 'chr3:1000-1234' in this format, and returns the length of the region

    start, end = region.split(':')[1].split('-')

    return int(end) - int(start)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python deconvolutionScript <input_mixtures.json> <output_predictions.json>")
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Chromosomes to keep (no sex chromosomes)
    acceptedChromosomes = set(['chr' + str(i) for i in range(1, 23)])

    # Load methylation counts and sample info
    sampleInfoPath = '/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/data/atlas-samples-overview.csv'  # atlas with cell type info etc
    outputPath = '/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/data/'
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

    # Build atlasMeans and atlasSigmas
    atlasMeans = np.zeros((len(allCellTypes), markerInd.shape[0]))
    atlasSigmas = np.zeros_like(atlasMeans)

    for j, ct in enumerate(allCellTypes):
        x = vstPerCt[ct].iloc[:, markerInd]
        atlasMeans[j] = x.mean()
        atlasSigmas[j] = x.std()

    # Load fake mixtures
    with open(input_json_path, 'r') as f:
        fake_mixtures_data = json.load(f)

    fake_df = pd.DataFrame(fake_mixtures_data).T
    fake_markers = fake_df.iloc[:, markerInd]

    # Cosine distance and prediction
    D = cdist(atlasMeans, fake_markers.values, metric='cosine')
    predicted_cts = [allCellTypes[np.argmin(D[:, i])] for i in range(D.shape[1])]

    # Output predictions to JSON
    predictions = {sample_name: predicted_cts[i] for i, sample_name in enumerate(fake_markers.index)}

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Also print predictions
    for name, pred in predictions.items():
        print(f'Sample {name} predicted as cell type: {pred}')

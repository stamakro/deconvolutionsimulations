import sys
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import cdist
import matlab.engine


def geneLength(region):
    # Extract chromosome region length from string like "chr1:100-200"
    try:
        coords = region.split(':')[1]
        start, end = map(int, coords.split('-'))
        return end - start
    except:
        return 1  # Avoid divide-by-zero or malformed region strings

def findMarkersTstat(vstPerCt, minTstat):
    allCellTypes = list(vstPerCt.keys())
    markers_hyper = {}
    markers_hypo = {}
    all_data = pd.concat(vstPerCt.values(), axis=0)
    num_regions = all_data.shape[1]

    for ct in allCellTypes:
        group = vstPerCt[ct]
        others = pd.concat([v for k, v in vstPerCt.items() if k != ct])
        mean_diff = group.mean().values - others.mean().values
        std_combined = np.sqrt(group.var().values / group.shape[0] + others.var().values / others.shape[0])
        tstats = np.divide(mean_diff, std_combined, out=np.zeros_like(mean_diff), where=std_combined!=0)
        markers_hyper[ct] = np.where(tstats >= minTstat)[0]
        markers_hypo[ct] = np.where(tstats <= -minTstat)[0]

    return {'hyper': markers_hyper, 'hypo': markers_hypo}


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



if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python deconvolutionScript.py <input_mixtures.csv> <output_predictions.csv>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    acceptedChromosomes = set(['chr' + str(i) for i in range(1, 23)])
    sampleInfoPath = '/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/data/atlas-samples-overview.csv'
    outputPath = '/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/data/'
                  
    data = pd.read_csv(outputPath + 'counts_aggregated_clusters.csv', index_col=0).T
    sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=';')
    if sampleInfo.shape[1] == 0:
        sampleInfo = pd.read_csv(sampleInfoPath, index_col=0, delimiter=',')

    rd = renameSample(sampleInfo.index)
    sampleInfo.rename(index=rd, inplace=True)
    sampleInfo = sampleInfo.loc[data.index]
    assert (sampleInfo.index == data.index).all()

    bigDF = pd.concat((data, sampleInfo), axis=1, ignore_index=False)
    bigDF = bigDF[bigDF['CpG/Total'] > 20.]
    bigDF = bigDF[bigDF['Used reads'] > 3000000]

    region_cols = bigDF.columns[pd.Series(bigDF.columns).apply(lambda x: x.split(':')[0]).isin(acceptedChromosomes)]
    sampleCountsAutosomal = bigDF[region_cols]

    countsAutosomal = np.array(sampleCountsAutosomal, int)
    lengths = np.array(pd.Series(sampleCountsAutosomal.columns).apply(geneLength))
    countsPerBase = countsAutosomal / lengths
    total = np.sum(countsPerBase, axis=1)
    tpm = 1e6 * (countsPerBase.T / total).T
    logtpm = np.log(tpm + 1)
    assert not np.sum(np.isnan(logtpm))

    normalizedData = pd.DataFrame(data=logtpm, index=sampleCountsAutosomal.index, columns=sampleCountsAutosomal.columns)
    allCellTypes = sorted(bigDF['Cell type'].unique())

    cellTypePresenceDraft = bigDF.groupby('Cell type').count().iloc[:, 0].to_dict()
    celltypePresence = np.array([cellTypePresenceDraft[ct] for ct in allCellTypes])

    vstPerCt = {ct: normalizedData.iloc[np.where(bigDF['Cell type'] == ct)[0]] for ct in allCellTypes}

    minTstat = 1.5
    allMarkers = findMarkersTstat(vstPerCt, minTstat)

    groups = bigDF.groupby('Cell type').groups
    newInd = []
    for ct in allCellTypes:
        newInd += list(np.where(bigDF['Cell type'] == ct)[0])
    bigDF = bigDF.iloc[newInd]
    normalizedData = normalizedData.iloc[newInd]

    markerHyperInd = np.hstack([v for _, v in allMarkers['hyper'].items()])
    markerHypoInd = np.hstack([v for _, v in allMarkers['hypo'].items()])
    markerInd = np.hstack((markerHyperInd, markerHypoInd))

    atlasMeans = np.zeros((len(allCellTypes), markerInd.shape[0]))
    atlasSigmas = np.zeros_like(atlasMeans)

    for j, ct in enumerate(allCellTypes):
        x = vstPerCt[ct].iloc[:, markerInd]
        atlasMeans[j] = x.mean()
        atlasSigmas[j] = x.std()

    # Read mixtures csv
    fake_df = pd.read_csv(input_csv_path, index_col=0).T
    marker_cols = sampleCountsAutosomal.columns[markerInd]
    fake_markers = fake_df.loc[marker_cols, :]

    # Start MATLAB and run deconvolve
    eng = matlab.engine.start_matlab()
    eng.cd(r'/projects/0/AdamsLab/Scripts/afroditi/deconvolutionsimulations/src/matlab', nargout=0)

    predicted_cts = []
    for i in range(fake_markers.shape[0]):
        sample = fake_markers.iloc[i].values.tolist()
        profile_matlab = matlab.double(sample)
        atlas_matlab = matlab.double(atlasMeans.tolist())
        res = eng.deconvolve(profile_matlab, atlas_matlab, 1.0, 0.1)
        res_np = np.array(res).flatten()
        top_idx = np.argmax(res_np)
        predicted_cts.append(allCellTypes[top_idx])

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        "Sample": fake_markers.index,
        "PredictedCellType": predicted_cts
    })
    predictions_df.to_csv(output_csv_path, index=False)

    for name, pred in zip(fake_markers.index, predictedCT):
        print(f'Sample {name} predicted as cell type: {pred}')


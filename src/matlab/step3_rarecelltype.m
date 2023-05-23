clear all;

rng(120621);


config = ReadYaml('configs/test.yaml');
% parameters of simulation
maxNumTissues = config.maxNumTissues;
NtissueSamples = config.NtissueSamples;
Ngenes = config.Ngenes;
snoise = config.noise_sigma;

% first half cases, 2nd half controls
Nsamples = config.Nsamples;
Ncases = fix(Nsamples / 2);
% proportion of rare cell type
rareP = config.rarectFrac;

minNumTissues = config.minNumTissues;

Niters = maxNumTissues - minNumTissues + 1;

estimatedProportionsCases = zeros(Niters, Ncases);
estimatedProportionsCtrls = zeros(Niters, Ncases);

for Ntissues = minNumTissues:maxNumTissues
    fprintf('%d', Ntissues);
    % generate mean and variance for each tissue
    mus = normrnd(0, 1, Ntissues, Ngenes);
    sigmas = exprnd(0.5, Ntissues, Ngenes);
    
    % generate atlas data, estimate parameters
    referenceData = zeros(NtissueSamples * Ntissues, Ngenes);
    labels = zeros(NtissueSamples * Ntissues, 1);
    estimatedMus = zeros(size(mus));
    estimatedSigmas = zeros(size(mus));
    
    for i = 1:Ntissues
        rr = normrnd(repmat(mus(i,:), [NtissueSamples,1]), repmat(sigmas(i,:), [NtissueSamples,1]));
        referenceData(1+NtissueSamples*(i-1):NtissueSamples*i,:) = rr;
        labels(1+NtissueSamples*(i-1):NtissueSamples*i) = i;
    
        estimatedMus(i,:) = mean(rr);
        estimatedSigmas(i,:) = std(rr);
    end
    
    % ground truth proportions for these samples
    trueFractions = zeros(Nsamples, Ntissues);
    trueFractions(1:Ncases, 1) = rareP;

    trueFractions(1:Ncases, 2:end) =  (1 - rareP) * drchrnd(ones(1,Ntissues-1), Ncases);
    trueFractions(Ncases+1:end, 2:end) =  drchrnd(ones(1,Ntissues-1), Ncases);
    
    % generate cfDNA samples
    data = trueFractions * normrnd(mus,sigmas);
    % data = trueFractions * mus;
    if snoise > 0
        data = data + normrnd(0, snoise, [Nsamples, Ngenes]);
    end

    for i = 1:Ncases
        estimatedFractions = deconvolve(data(i,:),estimatedMus, config.C, config.epsilon);
        estimatedProportionsCases(Ntissues - minNumTissues + 1, i) = estimatedFractions(1);
    end

    for i = 1:Ncases
        estimatedFractions = deconvolve(data(Ncases + i,:),estimatedMus, config.C, config.epsilon);
        estimatedProportionsCtrls(Ntissues - minNumTissues + 1, i) = estimatedFractions(1);
    end

end

folder = config.folder;
path = strcat('results/', folder, '/', num2str(rareP), '/');

mkdir(path);

writematrix(estimatedProportionsCtrls, strcat(path, 'res1ctrl.csv'));
writematrix(estimatedProportionsCases, strcat(path, 'res1case.csv'));


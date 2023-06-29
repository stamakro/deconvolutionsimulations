clear all;

rng(120621);

% parameters of simulation
maxNumTissues = 10;
Ntissues = 3;
NtissueSamples = 5;
Ngenes = 50;
snoise = 0.1;
Nsamples = 200;

minNumTissues = 3;

Niters = maxNumTissues - minNumTissues + 1;
Cs = zeros(Niters, Nsamples);
mses = zeros(Niters, Nsamples);
maes = zeros(Niters, Nsamples);
maxDevs = zeros(Niters, Nsamples);

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
    trueFractions = drchrnd(ones(1,Ntissues), Nsamples);
    
    % generate cfDNA samples
    data = trueFractions * normrnd(mus,sigmas);
    % data = trueFractions * mus;
    if snoise > 0
        data = data + normrnd(0, snoise, [Nsamples, Ngenes]);
    end

    for i = 1:Nsamples

        %estimatedFractions = deconvolve(data(i,:),estimatedMus, 1.0, 0.1);
        estimatedFractions = deconvolveLsq(data(i,:),estimatedMus, 1.0);
        [Cs(Ntissues - minNumTissues + 1, i), mses(Ntissues - minNumTissues + 1, i), maes(Ntissues - minNumTissues + 1, i), maxDevs(Ntissues - minNumTissues + 1, i)] = evaluateTissueOfOrigin(trueFractions(i,:)', estimatedFractions);
    end
end

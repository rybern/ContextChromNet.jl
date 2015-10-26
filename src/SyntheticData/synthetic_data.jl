module synthetic_data

export synthMVG, synthFilepath, guessMVGMoments, randCov, sparsify, zeroRandOffDiag, MVGSynthSpec, MVGModelSpec, initMVG, randAssignments, initLabelModel, GammaModelSpec, CentroidModelSpec, initBWModel, defaultStateSpec, defaultCentroidSpec, defaultGammaSpec, defaultSynthSpec, defaultProblemSpec

using Distributions
using mix_glasso2
using StatsBase
using hmm_types

import Base.hash

function defaultProblemSpec(; n=10000, p=30, k=4)
    ProblemSpec(n, p, k)
end

### Random MVG model

immutable MVGModelSpec
    avgRange :: Float64
    sparsity :: Float64
    dwellProb :: Float64
end

function defaultStateSpec(spec = defaultProblemSpec(); 
                          avgRange = 1,
                          sparsity = .5,
                          dwellProb = 3/4)
    MVGModelSpec(avgRange, sparsity, dwellProb)
end

function initBWModel(spec, modelSpec :: MVGModelSpec; load = true, save = true, seed = 0)
    initMVG(spec, modelSpec, load=load, save=save, seed=seed)
end

function hash(x :: MVGModelSpec)
    hash((x.avgRange, x.sparsity, x.dwellProb))
end

function initMVG(spec, modelSpec :: MVGModelSpec; load=true, save=true, seed = 0)
    params = (spec, modelSpec)
    diskMemoize(load, save, modelMVG, "modelMVG", params, seed)
end

function modelMVG(spec, modelSpec :: MVGModelSpec)
    dists = Array(MvNormal, spec.k)
    for i = 1:spec.k
        mu = rand(spec.p) * modelSpec.avgRange - modelSpec.avgRange/2;
        cov = randCov(spec.p, modelSpec.sparsity)
        
        dists[i] = MvNormal(mu, cov)
    end 

    states = map(dist -> HMMState(dist, true), dists)

    HMMStateModel(states, uniformTransMatrix(spec.k, modelSpec.dwellProb))
end

### Random assignment model

immutable GammaModelSpec
    dwellProb :: Float64
end

function defaultGammaSpec(spec = defaultProblemSpec();
                          dwellProb = 1/spec.k)
    GammaModelSpec(dwellProb)
end

function initBWModel(spec, modelSpec :: GammaModelSpec; load = true, save = true, seed = 0)
    initLabelModel(spec, modelSpec, load=load, save=save, seed=seed)
end

function hash(x :: GammaModelSpec)
    hash(x.dwellProb)
end

function initLabelModel(spec, gammaSpec :: GammaModelSpec; load=true, save=true, seed = 0)
    params = (spec, gammaSpec)
    diskMemoize(load, save, randGamma, "randGamma", params, seed)
end

function randGamma(spec, gammaSpec)
    trans = uniformTransMatrix(spec.k, gammaSpec.dwellProb)
    initAssignments = randAssignments(spec, trans, spec.n)
    initGamma = zeros(spec.k, spec.n);
    for i = 1:spec.n
        initGamma[initAssignments[i], i] = 1.0;
    end
    HMMLabelModel(initGamma, uniformTransMatrix(spec.k, gammaSpec.dwellProb))
end

### Random centroid model

immutable CentroidModelSpec
    numPoints :: Int64
end

function defaultCentroidSpec(spec = defaultProblemSpec();
                             numPoints = 2 * spec.k)
    CentroidModelSpec(numPoints)
end

function initBWModel(spec, modelSpec :: CentroidModelSpec; load = true, save = true, seed = 0)
    initCentroids(spec, modelSpec, load=load, save=save, seed=seed)
end

function hash(x :: CentroidModelSpec)
    hash(x.numPoints);
end

function initCentroids(spec, modelSpec :: CentroidModelSpec; load=true, save=true, seed = 0)
    params = (spec, modelSpec)
    diskMemoize(load, save, randCentroids, "randCentroids", params, seed)
end

function randCentroids(spec, centroidSpec)
    HMMCentroidModel(sample(1:spec.n, centroidSpec.numPoints, replace=false));
end

### Random synthetic data

immutable MVGSynthSpec
    modelSpec :: MVGModelSpec
    init :: Int64
end

function defaultSynthSpec(spec = defaultProblemSpec();
                          modelSpec = defaultStateSpec(spec, dwellProb = 3/4),
                          init = 1)
    MVGSynthSpec(modelSpec, init)
end

function hash(x :: MVGSynthSpec)
    hash((x.init, hash(x.modelSpec)))
end

function synthMVG(spec, synthSpec :: MVGSynthSpec; load=true, save=true, seed = 0)
    params = (spec, synthSpec)
    diskMemoize(load, save, generateMVG, "generateMVG", params, seed)
end

function generateMVG(spec, synthSpec :: MVGSynthSpec)
    truthModel = modelMVG(spec, synthSpec.modelSpec)

    (data, assignments) = generate(spec, truthModel, glSample, synthSpec.init)

    (data, assignments, truthModel)
end

### Utilities

function hash(x :: ProblemSpec)
    hash((x.n, x.p, x.k))
end


function uniformTransMatrix(k, prop)
    eye(k) * prop + ((1 - prop) / (k - 1)) * (ones(k, k) - eye(k));
end

#found in GaussianMixtures, should produce a PSD matrix
function randInvcov(p)
    T = rand(p, p)
    inv(cholfact(T' * T / p))
end

function randCov(p)
    inv(cholfact(randInvcov(p)))
end

function randCov(p, sparsity)
    aproxs = [aproxRandCov(p, sparsity) for i = 1:30]
    m_sparsity = M -> length(filter(x -> x < 10e-8, inv(M))) / (p*p)
    distances = map(M -> abs(sparsity - m_sparsity(M)), aproxs)
    aproxs[indmin(distances)]
end

function aproxRandCov(p, sparsity)
    generator = () -> rand(p, p)
    p_sparsity = 1-sqrt((1-sparsity) / p)
    P = sparsifyRand(generator, p_sparsity, X -> rank(X) == p)
    B = P' * P
    M = inv(cholfact(B))
    M / mean(abs(M))
end

function sparsifyRand(generator, sparsity, valid = isposdef)
    M = generator()

    s = size(M)[1] * size(M)[2];
    nzeros = s * sparsity;

    for i = 1:nzeros/2
        M = zeroRandOffDiag(M, valid);
        if(M == false)
            return sparsifyRand(generator, sparsity);
        end
    end

    M
end

function zeroRandOffDiag(M, valid = isposdef)
    (n, m) = size(M);

    for i = shuffle([1:n])
        for j = shuffle([1:(i-1)])
            if (M[i, j] == 0)
                continue
            end

            temp = M[i, j]

            M[i, j] = 0.0;
            M[j, i] = 0.0;

            if(valid(M))
                return M
            else
                M[i, j] = temp
                M[j, i] = temp
            end
        end
    end

    return false
end

function generate(spec, truthModel, sample, init)
    labels = randAssignments(spec, truthModel.trans, spec.n, init)

    data = Array(Float64, spec.p, spec.n)

    for i = 1:spec.n
        data[:, i] = sample(truthModel.states[labels[i]]);
    end

    (data, labels)
end

function randAssignments(spec, trans, n, init = 1)
    dists = Array(Categorical, spec.k)
    for i = 1:spec.k
        dists[i] = Categorical(vec(trans[i, :]))
    end

    labels = Array(Int32, spec.n)

    state = init
    for i = 1:spec.n
        labels[i] = state;
        state = rand(dists[state]);
    end

    labels
end

function memoFilepath(params...; basedir = "saved_memos")
    if(!isdir(basedir))
        mkdir(basedir)
    end
    paramHash = hashTuple(params...);
    "$basedir/$paramHash.memo";
end

function diskMemoize(load, save, fn, fnID, params, seed = 0)
    filename = false

    if(load)
        if(filename == false)
            filename = memoFilepath(fnID, seed, params...)
        end

        if(isfile(filename))
            print("found function result ($fnID) at $filename\n")

            inStream = open(filename, "r");
            data = deserialize(inStream);
            close(inStream)

            return data
        else
            print("no function result ($fnID) at $filename\n")
        end 
    end

    data = fn(params...)
    if (save)
        if(filename == false)
            filename = memoFilepath(fnID, seed, params...)
        end
            
        outStream = open(filename, "w")
        serialize(outStream, data)
        close(outStream)

        print("saved function result ($fnID) at $filename\n")
    end

    data
end

function hashTuple(tuple...)
    hashes = map(hash, tuple)

    reduce((x, y) -> hash(x + y), hashes)
end

end

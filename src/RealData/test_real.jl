module test_real

export estimateMixture, runPPIEval, plotPPIFromDumps, runLarge, runEval

using mix_glasso2
using synthetic_data
using hmm
using loading
using utils
using ppi_evaluation

using PyPlot

mixtureDump = "mixture.dump"
jointDump = "joint.dump"
covDump = "wholeCov.dump"
glassoDump = "wholeGlasso.dump"
estimateDump = "estimate.dump"
modelDump = "model.dump"
resultDump = "result.dump"

function runChromHMM(k, nIter, outputDir, baseDir = "ChromHMM/GSM646_binary_serialized/", binary_lengths_file = "binary_lengths.txt")
    if(outputDir != false && !isdir(outputDir))
        mkdir(outputDir)
    end

        lengths = readdlm(binary_lengths_file, ' ')
        println(lengths)
        lengths = lengths[:,2]
    data = convert(Array{Bool}, open(deserialize, "$baseDir/data.serial"))
    header = loadHeader("$baseDir/header")
    celltypes = loadHeader("$baseDir/celltypes")

    n_celltypes = length(celltypes)
    celltype_data_length = size(data, 2) / n_celltypes
    println("celltype_data_length = $celltype_data_length")
    celltype_data_length = int(celltype_data_length)

        println(lengths)
        
    celltype_data = [data[:, (((i-1) * celltype_data_length) + sum(lengths[1:j-1]) + 1) : (((i-1) * celltype_data_length) + sum(lengths[1:j]))]
                     for i = 1:n_celltypes, j = 1:length(lengths)]

    initSpec = defaultGammaSpec()
    update = buildWeightedFits(fitFullCov)
    emission = buildStateEmission(glLogEmission)

    celltype_results = [simpleBW(data_chunk, k, nIter, initSpec = initSpec, update = update, emission = emission) for data_chunk = celltype_data]

    dump(celltype_results, maybeCat(outputDir, resultDump))
    dump(estimate, maybeCat(outputDir, estimateDump))
end

function runLarge(outputDir)
    withDataHeader((data, header) -> runPPIEval(data, header, outputDir = outputDir), 0)
end

function runEval(n = 2000, nIter = 1, k = 2)
    withDataHeader((data, header) -> runPPIEval(data, header, k,
                                                bw = (data, k) -> simpleBW(data, k, nIter)), n)
end

function runPPIEval(data, header, k = 3;
                    outputDir = false,
                    bw = simpleBW,
                    averagedTargets = false)
    if(outputDir != false && !isdir(outputDir))
        mkdir(outputDir)
    end

    cov = wholeCov(data)
#    glasso = wholeGlasso(data)
    (estimate, states) = bw(data, k)
    mixture = modelToMixture(states)
    joint = composeNetsByMax(mixture...)

    dump(mixture, maybeCat(outputDir, mixtureDump))
    dump(joint, maybeCat(outputDir, jointDump))
    dump(cov, maybeCat(outputDir, covDump))
 #   dump(glasso, maybeCat(outputDir, glassoDump))
    dump(estimate, maybeCat(outputDir, estimateDump))

    plotMixturePPI(mixture, cov, header, joint,
                   dumpfile = maybeCat(outputDir, "PPI_Plot_k$(k).png"),
                   averagedTargets = averagedTargets)

    mixture, joint, cov, glasso
end

function plotPPIFromDumps(header, dumpDir; figureFile = false, details = false)
    mixture = scoop("$(dumpDir)$mixtureDump")        
    joint = scoop("$(dumpDir)$jointDump")
    whole = scoop("$(dumpDir)$covDump")

    plotMixturePPI(mixture, whole, header, joint, dumpfile = figureFile, details = details)
end

function barPlotPPIFromDumps(header, dumpDir; figureFile = false)
    mixture = scoop("$(dumpDir)$mixtureDump")        
    #joint = scoop("$(dumpDir)$jointDump")
    whole = scoop("largeAll_k3/$covDump")

    plotBarMixturePPI(mixture, whole, header, dumpfile = figureFile)
end

function wholeCov(data)
    c = cov(data, vardim = 2)

    inv(c) # SHOULD be posdef, but sometimes isn't. woooo
end

function wholeGlasso(data)
    cov(fitGLasso(data, ones(size(data, 2))))
end

end 

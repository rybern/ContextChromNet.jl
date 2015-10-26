module test_synth

export gammaAssignments, stateMapping, avgSoftAssignmentDistance, testBW, syntheticValidation, testBWParams, hardAssignmentAccuracy, evalStatistics, repeatEvalStatistics, syntheticValidation, plotSet

using loading
using hmm

using ppi_evaluation
using hmm_types
using mix_glasso2
using synthetic_data
using lambda_optimization
using Distributions
using StatsBase
using barchart_plots


function testMStep(; spec = defaultProblemSpec(n = 50000),
                   synthSpec = defaultSynthSpec(spec, modelSpec = defaultStateSpec(spec, sparsity = .7, avgRange = 0)))
    (data, assignments, truth) = synthMVG(spec, synthSpec);

    train_size = div(spec.n, 2)
    train_is = 1 : train_size
    test_is = train_size + 1 : spec.n
    train_data = data[:, train_is]
    test_data = data[:, test_is]
    train_assignments = assignments[train_is]
    test_assignments = assignments[test_is]

    gamma = zeros(train_size, spec.k)
    for i = 1:train_size
        gamma[i, assignments[i]] = 1
    end

    adaptive_models = static_cov_mixture(data[:, train_is], gamma)
    static_models = static_glasso_mixture(data[:, train_is], gamma)
    truth_models = map(state -> state.dist, truth.states)

    models = (adaptive_models, static_models, truth_models)
    model_names = ("cov", "static", "truth")

    for i = i = 1:length(models)
        model = models[i]

        (test_accuracy, net_density, adj_accuracy, testll, trainll, hmm_test_ll, hmm_train_ll, network_enrichment) = validate_weighted_fit(spec, model, truth, train_data, train_assignments, test_data, test_assignments)

        println(model_names[i], " model:")
        println("label: $test_accuracy")
        println("network enrichment: $network_enrichment")
        println("network accuracy: $adj_accuracy")
        println("network density: $net_density")
        println("dist test ll: $testll")
        println("dist train lls: $trainll")
        println("hmm test ll: $hmm_test_ll")
        println("hmm train lls: $hmm_train_ll")
    end

    (adaptive_models, static_models, truth_models)
end

function validate_weighted_fit(spec, models, truth, train_data, train_assignments, test_data, test_assignments)
    max_num_edges = (spec.p ^ 2 - spec.p)
    
    gamma = hcat([logpdf(model, test_data) for model = models]...)
    test_predictions = [indmax(gamma[row, :]) for row = 1:size(gamma, 1)]
    test_accuracy = mean(map((==), test_predictions, test_assignments))
    net_density = mean([sum(networkNonzeroMatrix(models[i]) .* (1 - eye(spec.p))) / max_num_edges for i = 1:spec.k])
    adj_accuracy = mean([accuracy(networkNonzeroMatrix(truth.states[i].dist), networkNonzeroMatrix(models[i])) for i = 1:spec.k])

    testll = sum([loglikelihood(models[i], test_data[:, test_assignments .== i]) for i = 1:spec.k])
    trainll = sum([loglikelihood(models[i], train_data[:, train_assignments .== i]) for i = 1:spec.k])


    
    hmm_model = HMMStateModel([HMMState(model, true) for model = models], truth.trans)
    train_spec = ProblemSpec(size(train_data, 2), spec.p, spec.k)
    test_spec = ProblemSpec(size(test_data, 2), spec.p, spec.k)
    
    hmm_train_ll = logLikelihood(train_spec, train_data, hmm_model, buildStateEmission(glLogEmission))
    hmm_test_ll = logLikelihood(test_spec, test_data, hmm_model, buildStateEmission(glLogEmission))
    avg_network_enrichment = mean([networkTopEdgesAccuracy(truth.states[i].dist, models[i]) for i = 1:spec.k])

    [test_accuracy, net_density, adj_accuracy, testll, trainll, hmm_test_ll, hmm_train_ll, avg_network_enrichment]
end



function testBW(; spec = defaultProblemSpec(n = 500),
                initSpec = defaultGammaSpec(spec),
                synthSpec = defaultSynthSpec(spec, modelSpec = defaultStateSpec(spec, sparsity = .5)))

    println("start synth ", spec.n)
    (data, assignments, truth) = synthMVG(spec, synthSpec);
    println("done synth ", size(data))
    guessGenerator = seed -> initBWModel(spec, initSpec, seed = seed)

    updateStates = jointWeightedModel()
    emission = buildStateEmission(glLogEmission)

    println("starting")

    (estimate, model) = maxLikelihoodBW(spec, guessGenerator, 5, emission, updateStates, data)
    (orderedEstimate, orderedModel) = reorderStates(spec, model, estimate, assignments)

    (truth, orderedModel, assignments, orderedEstimate, data)
end

function synthValLine(res, stat, fit)
    map(arr -> arr[stat], res[:, fit])
end

function plot_standard_synth_bars(results)
    grouped_bar_plot([(results[1][1][1], results[1][1][2], "GLasso"),
                      (results[1][2][1], results[1][2][2], "Full Cov")], 
                     [.25:.25:1], 
                     "True model density", 
                     "Top-N Edge Network Accuracy", 
                     "Network quality vs. Density")

    grouped_bar_plot([(results[2][1][1], results[2][1][2], "GLasso"),
                      (results[2][2][1], results[2][2][2], "Full Cov"),
                      (results[2][3][1], results[2][3][2], "Diag Cov")],
                     [.25:.25:1], 
                     "True model density", 
                     "Label Accuracy",
                     "Label Accuracy vs. Density",
                     bar_width = .3)
end

function syntheticValidation (densities = [.25:.25:1],
                              fitFunctions = [fitGLasso, fitFullCov, fitDiagCov];
                              spec = defaultProblemSpec(n = 50000),
                              synthSpec = defaultSynthSpec(spec),
                              initSpec = defaultGammaSpec(spec))

    function sparsitySynthSpec(density)
        MVGSynthSpec(MVGModelSpec(synthSpec.modelSpec.avgRange,
                                  1-density,
                                  synthSpec.modelSpec.dwellProb)
                     , synthSpec.init)
    end

    train_statistics = [averageNetworkTopEdgesStatistic,
                        hardAssignmentStatistic]
    test_statistics = []
    n_statistics = length(train_statistics) + length(test_statistics)

    results = [repeatEvalStatistics(5,
                          train_statistics,
                          test_statistics,
                          1,
                          buildStateEmission(glLogEmission),
                          buildWeightedFits(fitFunction),
                          spec,
                          sparsitySynthSpec(density),
                          initSpec) for density = densities, fitFunction = fitFunctions]

    [[([results[density_i, model_i][1][statistic_i] for density_i = 1:length(densities)], 
       [results[density_i, model_i][2][statistic_i] for density_i = 1:length(densities)])
      for model_i = 1:length(fitFunctions)]
     for statistic_i = 1:n_statistics]
end

function repeatEvalStatistics (numRepeats, args...)
    runs = [evalStatistics(args..., seed = i) for i = 1:numRepeats]
#    numeric_collection_moments(runs)
    runs
end

function numeric_collection_moments(coll)
    stds = [std(float([coll[i][j] for i = 1:length(coll)])) for j = 1:length(coll[1])]
    means = mean(coll)
    (means, stds)
end

function evalStatistics (trainStatistics,
                         testStatistics,
                         testProportion,
                         emission,
                         update,
                         spec = defaultProblemSpec(),
                         synthSpec = defaultSynthSpec(spec),
                         initSpec = defaultCentroidSpec(spec);
                         seed = 0)

    testSpec = ProblemSpec(spec.n * testProportion,
                           spec.p, 
                           spec.k);
    synthProbSpec = ProblemSpec(spec.n * (1 + testProportion),
                                spec.p, 
                                spec.k);

    (data, assignments, truth) = synthMVG(synthProbSpec, synthSpec, seed = seed);
    guessGenerator = seed_ -> initBWModel(spec, initSpec, seed = hash(seed) + hash(seed_))

    dataTrain = data[:, 1 : spec.n]
    assignmentsTrain = assignments[1 : spec.n]

    dataTest = data[:, spec.n+1 : synthProbSpec.n]
    assignmentsTest = assignments[spec.n+1 : synthProbSpec.n]
    
    (unorderedTrainEstimate, unorderedModel) = maxLikelihoodBW(spec, guessGenerator, 1, emission, update, dataTrain)
    (trainEstimate, model) = reorderStates(spec, unorderedModel, unorderedTrainEstimate, assignmentsTrain)

    [[trainStat(spec, model, truth, trainEstimate, assignmentsTrain, dataTrain) for trainStat = trainStatistics],
     [testStat(testSpec, model, emission, truth, dataTest, assignmentsTest) for testStat = testStatistics]]
end

# Network structure
function averageNetworkNonzeroStatistic(spec, model, truth, trainEstimate, assignmentsTrain, dataTrain)
    averageNetworkNonzeroAccuracy(truth.states, model.states)
end

function averageNetworkNonzeroAccuracy(real, found)
    mean(map(adjacencyAccuracy, real, found))
end

function networkNonzeroMatrix(state)
    abs(inv(cov(state))) .> .01
end

function networkNonzeroAccuracy(real, found)
    if(found.active == true)
        accuracy(networkNonzeroMatrix(real.dist), networkNonzeroMatrix(found.dist))
    else
        0
    end
end

# Top edge network structure

function averageNetworkTopEdgesStatistic(spec, model, truth, trainEstimate, assignmentsTrain, dataTrain)
    averageNetworkTopEdgesAccuracy(truth.states, model.states)
end

function averageNetworkTopEdgesAccuracy(real, found)
    mean(map((r, f) -> f.active ? networkTopEdgesAccuracy(r.dist, f.dist) : 0, real, found))
end

function networkTopEdgesMatrix(state)
    abs(inv(cov(state))) .> .01
end

function networkTopEdgesAccuracy(real, found)
    weighted_true_edges = sortedEdges(inv(cov(real)))
    
    true_edges = map(t -> t[2], filter(edge -> abs(edge[1]) > 1e-7, weighted_true_edges)) 
    n_true_edges = length(true_edges)
    
    weighted_model_edges = sortedEdges(inv(cov(found)))
    model_edges = map(t -> t[2], filter(edge -> abs(edge[1]) > 1e-7, weighted_model_edges[1:n_true_edges]))
    
    length(intersect(Set(model_edges), Set(true_edges))) / n_true_edges
end


# Label assignment

function hardAssignmentStatistic(spec, model, truth, trainEstimate, assignmentsTrain, dataTrain)
    hardAssignmentAccuracy(assignmentsTrain, trainEstimate.gamma)
end

function hardAssignmentAccuracy(assignments, gamma)
    guesses = gammaAssignments(gamma)
    accuracy(guesses, assignments)
end


# Likelihood

function testLogLikelihoodStatistic(spec, model, emission, truth, dataTest, assignmentsTest)
    logLikelihood(spec, dataTest, model, emission)
end

# could be done lazily if performance is an issue
function reorderStates(spec, model, estimate, assignments)
    mapping = stateMapping(assignments, estimate.gamma)

    newStates = [model.states[mapping[i]] for i = [1:spec.k]]
    newTrans = Array(Float64, spec.k, spec.k)
    for i = 1:spec.k
        ip = mapping[i]
        for j = 1:spec.k
            jp = mapping[j]
            newTrans[i, j] = model.trans[ip, jp]
        end
    end
    newModel = HMMStateModel(newStates, newTrans)

    newInit = [estimate.init[mapping[i]] for i = [1:spec.k]]
    newGamma = Array(Float64, spec.n, spec.k)
    for i = 1:spec.k
        newGamma[:, i] = estimate.gamma[mapping[i], :];
    end
    newEstimate = HMMEstimate(newInit, newGamma)

    (newEstimate, newModel)
end

# training structure accuracy
# test log-likelihood
# test label accuracy

function precisionRecall(truth, found)
    l = length(truth)
    tp = 0
    fn = 0
    fp = 0

    for i = 1:l
        if(found[i] && truth[i])
            tp = tp + 1;
        elseif(!found[i] && truth[i])
            fn = fn + 1;
        elseif(found[i] && !truth[i])
            fp = fp + 1;
        end
    end
        
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)

    (prec, rec)
end

function accuracy(truth, found)
    count(identity, truth .== found) / length(truth)
end
# find the best permutation real -> found
function stateMapping(real, gamma)
    (k, n) = size(gamma)
    perms = collect(permutations([1:k]))
    
    function permError(perm)
        foldl((total, i) -> total + 1 - gamma[perm[real[i]], i], 0, [1:n])
    end
        
    minPermInd = indmin(map(permError, perms))
    perms[minPermInd]
end

function avgSoftAssignmentDistance(real, gamma; mapping=stateMapping(real, gamma))
    n = size(real)[1]
    foldl((total, i) -> total + 1 - gamma[i, findfirst(mapping, real[i])], 0, [1:n]) / n
end

function avgSoftAssignmentDistance(res)
    avgSoftAssignmentDistance(res.assignments, res.estimate.gamma)
end

function gammaAssignments(gamma)
    [indmax(gamma[i, :]) for i = 1:size(gamma, 1)]
end


using PyPlot

function plotSynthVal(res;
                      statNames = ["Adjacency Accuracy", "Assignment Accuracy", "Test Log-Likelihood"],
                      fitNames = ["GLasso", "Full Cov", "Diag Cov"],
                      sparsities = [0:.05:.75])
    
    for i in 1:size(statNames)[1]
        plotSynthValStat(res, i, [1:3], sparsities, statNames[i], fitNames, i)
    end
end

function plotSynthValStat(res, stat, fits, sparsities, statName, fitNames, fignum)
    x = sparsities
    
    figure(fignum)

    for i = 1:size(fits)[1]
        y = synthValLine(res, stat, fits[i])
        plot(x, y, label = fitNames[i])
    end
    
    xlabel("Sparsity")
    ylabel(statName)
    title("$statName vs. Sparsity")
    legend()
    println("showing")
    show()
end

end

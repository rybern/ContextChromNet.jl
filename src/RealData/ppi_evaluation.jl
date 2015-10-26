module ppi_evaluation

export labeledSortedEdges, initTruthMatching, plotPPI, plotMixturePPI, composeNetsByMax, sortedEdges, plotBarMixturePPI

using loading
using evaluation
using PyPlot
using barchart_plots

# states -> plot
function mixturePPI(mixture,
                    n,
                    experimentPredicate = x -> x["cellType"] == "K562")
    (header, data) = loadFiltered(experimentPredicate, firstN = n)
    whole = inv(cholfact(cov(loading.baToFloat(data))))

    plotMixturePPI(mixture, whole)
end

function enrichmentFold(network,
                        header,
                        pairs;
                        averagedTargets = false)
    (p, r) = networkPR(network, header, averagedTargets, pairs)
    println(p[1:27])
    n_possible_edges = length(p)
    n_true_edges = int(last(p) * n_possible_edges)
    println("poss $n_possible_edges true $n_true_edges")
    println("n pairs:", length(pairs))
    
    n_top_correct = p[n_true_edges] * n_true_edges
    println(n_top_correct)
    random_correct = n_true_edges * n_true_edges / n_possible_edges
    println("rand $random_correct")

    n_top_correct / random_correct
end

# networks -> plot
function plotMixturePPI(mixture,
                        cov,
                        header,
                        joint = composeNetsByMax(mixture...);
                        dumpfile = false,
                        averagedTargets = false,
                        details = false)
    mixtureNames = ["State $i" for i = 1:size(mixture)[1]]
    nets = Array{Float64, 2}[cov, joint, mixture...]
    names = String["Whole InvCov"; "Max magnitude"; mixtureNames...]
    
    plotPPI(nets, names, header, dumpfile = dumpfile, averagedTargets = averagedTargets, details = details)
end

function plotBarMixturePPI(mixture,
                           cov,
                           header;
                           dumpfile = false,
                           averagedTargets = false,
                           details = false)
    mixtureNames = ["State $i" for i = 1:size(mixture)[1]]
    nets = Array{Float64, 2}[cov, mixture...]
    names = String["Whole InvCov"; mixtureNames...]
    
    plotBarPPI(nets, names, header, dumpfile = dumpfile, averagedTargets = averagedTargets, details = details)
end

function networkMatches(network, header = loading.loadFilteredHeader(), pairs = loading.loadPairs(),
                        averagedTargets = false)
    labelledEdges = labeledSortedEdges(network, header)

    edges = formatEdges([l[2] for l = labelledEdges], averagedTargets)

#    println(edges[1:15])

    truthMatching(edges, pairs)
end

function networkPR(network, header = loading.loadFilteredHeader(), pairs = loading.loadPairs(),
                   averagedTargets = false)
    prediction_truth = networkMatches(network, header, averagedTargets, pairs)
    prCurve(prediction_truth)
end

# networks -> names -> header -> plot
function plotPPI(networks, networkNames, header = loading.loadFilteredHeader(); 
                 dumpfile = false, averagedTargets = false, details = false)

    pairs = loadPairs()
    prs = [networkPR(network, header, averagedTargets, pairs) for network = networks]

    if(any(pr -> size(pr[1], 1) == 0, prs))
        println("one or more state makes no ppi predictions!")
        good_is = filter(i -> size(prs[i][1], 1) != 0, 1:size(networks, 1))
        prs = prs[good_is]
        networkNames[good_is]
    end
        

    for i = 1:size(prs, 1)
#        plot(prs[i][2], prs[i][1], label = networkNames[i])
#        plot(1:length(prs[i][1]), prs[i][1], label = networkNames[i])
        if (i==1)
            plot(prs[i][2]*(length(prs[i][2]) / length(prs[1][2])), prs[i][1], label = networkNames[i], linewidth = 20)
        else
            plot(prs[i][2]*(length(prs[i][2]) / length(prs[1][2])), prs[i][1], label = networkNames[i])
        end
    end

#    for i = 1:size(networks)[1]
#        (p, r) = networkPR(networks[i], header, averagedTargets)
#        n = size(p, 1)
#        plot([1:n], p, label = networkNames[i])
#    end

    ylabel("Precision")
    xlabel("Recall")

    name = "PPI Enrichment PR"
    if (details != false)
        name = string(name, " (", details, ")")
    end
    title(name)

    legend()
    show()

    if (dumpfile != false)
        savefig(dumpfile, bbox_inches="tight")
    end
end

function plotBarPPI(networks, networkNames, header = loading.loadFilteredHeader(); 
                    dumpfile = false, averagedTargets = false, details = false)

    pairs = loadPairs()
    enrichments = [enrichmentFold(network,
                                  header,
                                  pairs,
                                  averagedTargets = false) for network = networks]

    println("plotting?")
    basic_bar_plot(enrichments, networkNames, "State emission model", "Enrichment fold", "PPI Enrichment Folds")
end

# networks -> max network
function composeNetsByMax(nets...)
    maxMag = (a, b) -> abs(a) > abs(b) ? a : b
    maxMagArr = (ma, mb) -> map(maxMag, ma, mb)

    reduce(maxMagArr, nets)
end

# m -> header -> [(strength, (name, name))]
function labeledSortedEdges(m, header)
    edges = sortedEdges(m)
    
    labelPair = t -> (t[1], (header[t[2][1]], header[t[2][2]]))

    map(labelPair, edges)
end

# m -> [(strength, (ix1, ix2))]
function sortedEdges(m, magnitude=true)
    (a, b) = size(m);
    
    is = reshape([(m[i, j], (i, j)) for i=1:a, j=1:b], (a*b,))
    
    offDiags = filter(t -> t[2][1] < t[2][2], is)
    nonzeros = filter(t -> abs(t[1]) > 10.0^-10, offDiags)

    #TEMPORARY CHANGE!
    sorted = sort(offDiags,
                  by = t -> magnitude ? abs(t[1]) : t[1],
                  rev = true)

    sorted
end

function translateEdges(edges, mapping = loadMapping(reverse = false))
    mappingKeys = Set(keys(mapping))
    
    translatableEdges = filter(edge -> in(edge[1], mappingKeys) && in(edge[2], mappingKeys), edges)

    function translateGuess(pair)
        t = (mapping[pair[1]], mapping[pair[2]])
        t[1] > t[2] ? t : (t[2], t[1])
    end

    map(translateGuess, translatableEdges)
end

function validEdge(edge, targetBlacklist)
    if(edge[1] == edge[2])
        false
    elseif(in(edge[1], targetBlacklist) || in(edge[2], targetBlacklist))
        false
    else
        true
    end
end

function formatEdges(edges, averagedTargets = false; targetBlacklist = Set(["Q71DI3", "P0C0S5", "P62805", "P84243"]))
    if !averagedTargets
        edges = translateEdges(edges)
    end

    validEdges = filter(edge -> validEdge(edge, targetBlacklist), edges)

    unique(validEdges)
end

# IO ([(name, name)] -> [bool])
function truthMatching(edges, truthPairs = loadPairs())
    [in(edge, truthPairs) for edge = edges]
end

end


module EdgeUtils

export experiment_network_factor_edges, sorted_edges, network_enrichment

using Loading
using BaumWelchUtils

# network -> weighted indices (edges)
# header -> indices -> experiment pair
# header -> weighted indices -> weighted experiment pairs
# mapping -> experiment pair -> protien pair
# mapping -> weighted experiment pairs -> weighted protien pairs

function weighted_edges (network; eps = 1e-8, filter_small = false)
    (n, m) = size(network)
    
    edge_ixs = filter(t -> t[1] < t[2], [(i, j)
                                         for i = 1:n, j = 1:m])
    wixs = [(ix, network[ix[1], ix[2]])
            for ix = edge_ixs]

    if filter_small
        filter(t -> abs(t[2]) > eps, wixs)
    else
        wixs
    end
end

function sorted_edges (network; eps = 1e-8, filter_small = false)
    weighted = weighted_edges(network, eps = eps, filter_small = filter_small)
    map(t -> t[1], sort(weighted, by = t -> abs(t[2])))
end

function ix_to_exp_pair (header, ix)
    (header[ix[1]], header[ix[2]])
end

function weighted_ixs_to_weighted_exp_pairs (weighted_ixs,
                                             header)
    [(ix_to_exp_pair(header, wixs[1]), wixs[2])
     for wixs = weighted_ixs]
end

function exp_pair_to_factor_pair (mapping,
                                  exp_pair)
    f1 = mapping[exp_pair[1]]
    f2 = mapping[exp_pair[2]]

    f1 < f2 ? (f1, f2) : (f2, f1)
end

function weighted_exp_pairs_to_weighted_factor_pairs (weighted_exps,
                                                      mapping,
                                                      target_blacklist)
    mapping_keys = Set(keys(mapping))
    filter!(edge -> in(edge[1][1], mapping_keys) && in(edge[1][2], mapping_keys),
            weighted_exps)

    weighted_factor_pairs = [(exp_pair_to_factor_pair(mapping,
                                                      wexp[1]),
                              wexp[2])
                              for wexp = weighted_exps]

    filter!(edge -> edge[1][1] != edge[1][2] && !in(edge[1][1], target_blacklist) && !in(edge[1][2], target_blacklist),
            weighted_factor_pairs)

    sort!(weighted_factor_pairs,
          by = t -> abs(t[2]),
          rev = true)

    unique_by(weighted_factor_pairs,
              t -> t[1])
end

function experiment_network_factor_edges (network,
                                          header = load_filtered_header(),
                                          mapping = load_mapping(),
                                          target_blacklist = Set(["Q71DI3",
                                                                  "P0C0S5",
                                                                  "P62805",
                                                                  "P84243"]))
    wixs = weighted_edges(network)
    wexps = weighted_ixs_to_weighted_exp_pairs(wixs, header)
    wfs = weighted_exp_pairs_to_weighted_factor_pairs(wexps, mapping, target_blacklist)
end

# takes a list of edge prediction accuracy, sorted from most to least
# confident. Need the whole list to calculate the number of potential
# right answers (num_true)!
function enrichment (sorted_truth :: Array{Bool, 1})
    num_true = sum(sorted_truth)
    num_guesses = length(sorted_truth)
    num_correct = sum(sorted_truth[1:num_true])
    num_expected = num_true * num_true / num_guesses

    num_correct / num_expected
end
                     
function network_enrichment (network,
                             pairs = load_pairs(),
                             header = load_filtered_header(),
                             mapping = load_mapping(),
                             target_blacklist = Set(["Q71DI3",
                                                     "P0C0S5",
                                                     "P62805",
                                                     "P84243"]))
    weighted_edges = experiment_network_factor_edges (network, header, mapping, target_blacklist)
    weighted_edge_enrichment(weighted_edges,
                             pairs)
end

function weighted_edge_enrichment (weighted_edges,
                                   pairs = load_pairs())
    edge_truth = Bool[edge[1] in pairs
                      for edge in weighted_edges]
    enrichment (edge_truth)
end

function network_factor_density (network,
                                 header = load_filtered_header(),
                                 mapping = load_mapping(),
                                 target_blacklist = Set(["Q71DI3",
                                                         "P0C0S5",
                                                         "P62805",
                                                         "P84243"]);
                                 eps = 1e-8)
    weighted_edges = experiment_network_factor_edges (network,
                                                      header,
                                                      mapping,
                                                      target_blacklist)

    num_total = length(weighted_edges)
    num_claimed = length(filter(t -> abs(t[2]) > eps, weighted_edges))

    num_claimed / num_total
end

end


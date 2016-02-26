module ValidationMeasures
export join_measures, hard_label_accuracy_measure, hard_network_edge_accuracy_measure, network_enrichment_fold_measure, network_edge_matches_measure, train_loglikelihood_measure, test_loglikelihood_measure, whole_cov_data_measure, hard_network_edge_accuracy, hard_label_accuracy

using EdgeUtils
using BaumWelchUtils
using BaumWelch
using EmissionDistributions

function join_measures(measures)
    function joint_measure(data_train, labels_train,
                           data_holdout, labels_holdout,
                           true_model,
                           found_estimate, found_model, found_ll)
        [measure(data_train, labels_train,
                 data_holdout, labels_holdout,
                 true_model,
                 found_estimate, found_model, found_ll)
         for measure = measures]
    end
end

function hard_label_accuracy_measure(data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll)
    hard_label_accuracy(found_estimate.gamma, labels_train)
end

function hard_network_edge_accuracy_measure(data_train, labels_train,
                                            data_holdout, labels_holdout,
                                            true_model,
                                            found_estimate, found_model, found_ll)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    [hard_network_edge_accuracy(network_pair...)
     for network_pair = collect(zip(found_networks, true_networks))]
end

function train_loglikelihood_measure(data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll)
    found_ll
end

function test_loglikelihood_measure(data_train, labels_train,
                                    data_holdout, labels_holdout,
                                    true_model,
                                    found_estimate, found_model, found_ll)
    k = size(found_model.trans, 1)
    log_likelihood(data_holdout, k, found_model, dist_log_pdf)
end

function whole_cov_data_measure(data_train, labels_train,
                                data_holdout, labels_holdout,
                                true_model,
                                found_estimate, found_model, found_ll)
    cov(data)
end

function network_enrichment_fold_measure(data_train, labels_train,
                                         data_holdout, labels_holdout,
                                         true_model,
                                         found_estimate, found_model, found_ll)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    [network_enrichment_fold(network_pair...)
     for network_pair = collect(zip(found_networks, true_networks))]
end

function network_edge_matches_measure(data_train, labels_train,
                                      data_holdout, labels_holdout,
                                      true_model,
                                      found_estimate, found_model, found_ll)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    [network_edge_matches(network_pair...)[1]
     for network_pair = collect(zip(found_networks, true_networks))]
end

function network_edge_matches(found_network, true_network; eps = 1e-8)
    if found_network == Void
        return 0
    end

    true_edges = Set(sorted_edges(true_network, eps = eps, filter_small = true))
    num_true_edges = length(true_edges)

    all_found_edges = sorted_edges(found_network, eps = eps, filter_small = false)
    #all_found_edges = reverse(all_found_edges)
    num_found_edges = min(length(all_found_edges), num_true_edges)
    found_edges = all_found_edges[1:num_found_edges]

    (map(edge -> in((edge[1], edge[2]), true_edges) || in((edge[2], edge[1]), true_edges),
         found_edges),
     num_found_edges,
     num_true_edges)
end

function network_enrichment_fold(found_network, true_network; eps = 1e-8)
    if found_network == Void
        return 0
    end

    (truths, num_found_edges, num_true_edges) = network_edge_matches(found_network,
                                                                     true_network,
                                                                     eps = eps)


    found_true = num_found_edges == 0 ? 0 : sum(truths)
    num_possible = size(found_network, 1) * (size(found_network, 2) - 1) / 2
    random_true = num_true_edges * num_true_edges / num_possible

    found_true / random_true
end

function hard_label_accuracy(gamma, true_labels)
    found_labels = gamma_to_labels(gamma)
    accuracy(found_labels, true_labels)
end

function hard_network_edge_accuracy(found_mat, true_mat, eps = 1e-8)
    if found_mat == Void
        return 0
    end

    found_network = abs(found_mat) .> eps
    true_network = abs(true_mat) .> eps
    accuracy(found_network, true_network)
end

function accuracy{T <: Any}(vec1 :: AbstractArray{T},
                              vec2 :: AbstractArray{T})
    sum(map((==), vec1, vec2)) / length(vec1)
end

end

module ValidationMeasures
export join_measures, hard_label_accuracy_measure, num_active_states_measure, hard_network_edge_accuracy_measure, network_enrichment_fold_measure, network_edge_matches_measure, train_loglikelihood_measure, test_loglikelihood_measure, whole_cov_data_measure, hard_network_edge_accuracy, hard_label_accuracy

using EdgeUtils
using BaumWelchUtils
using BaumWelch
using EmissionDistributions
using StateMatching

function join_measures(measures)
    function joint_measure(data_train, labels_train,
                           data_holdout, labels_holdout,
                           true_model,
                           found_estimate, found_model, found_ll,
                           confusion_matrix = label_confusion_matrix(gamma_to_labels(found_estimate.gamma),
                                                                     size(found_model.trans, 1),
                                                                     labels_train,
                                                                     size(true_model.trans, 1)))
        [measure(data_train, labels_train,
                 data_holdout, labels_holdout,
                 true_model,
                 found_estimate, found_model, found_ll,
                 confusion_matrix)
         for measure = measures]
    end
end

function hard_label_accuracy_measure(data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll,
                                     confusion_matrix)
    # Permute the found gamma to best match the original labels
    new_to_old = StateMatching.label_permutation(confusion_matrix)
    permuted_gamma = StateMatching.permute_gamma(new_to_old, found_estimate.gamma)

    # Find label accuracy using the permuted gamma
    hard_label_accuracy(permuted_gamma, labels_train)
end

function hard_network_edge_accuracy_measure(data_train, labels_train,
                                            data_holdout, labels_holdout,
                                            true_model,
                                            found_estimate, found_model, found_ll,
                                            confusion_matrix)
    weighted_average_measures(hard_network_edge_accuracy,
                              found_model,
                              true_model,
                              confusion_matrix)
end

function train_loglikelihood_measure(data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll,
                                     confusion_matrix)
    found_ll
end

function test_loglikelihood_measure(data_train, labels_train,
                                    data_holdout, labels_holdout,
                                    true_model,
                                    found_estimate, found_model, found_ll,
                                    confusion_matrix)
    k = size(found_model.trans, 1)
    log_likelihood(data_holdout, k, found_model, dist_log_pdf)
end

function network_enrichment_fold_measure(data_train, labels_train,
                                         data_holdout, labels_holdout,
                                         true_model,
                                         found_estimate, found_model, found_ll,
                                         confusion_matrix)
    weighted_average_measures(network_enrichment_fold,
                              found_model,
                              true_model,
                              confusion_matrix)
end

function num_active_states_measure(data_train, labels_train,
                                   data_holdout, labels_holdout,
                                   true_model,
                                   found_estimate, found_model, found_ll,
                                   confusion_matrix)
    length(filter(state -> state.active, found_model.states))
end
function network_edge_matches_measure(data_train, labels_train,
                                      data_holdout, labels_holdout,
                                      true_model,
                                      found_estimate, found_model, found_ll,
                                      confusion_matrix)
    weighted_average_measures(network_edge_matches,
                              found_model,
                              true_model,
                              confusion_matrix,
                              join = zip)

end

function weighted_average_measures(pairwise_measure :: Function,
                                   found_model,
                                   true_model,
                                   confusion_matrix;
                                   join = dot)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    pairwise_matches = [pairwise_measure(found_network, true_network)
                        for found_network = found_networks, true_network = true_networks]

    weighted_matches = weighted_average_measures(pairwise_matches,
                                                 confusion_matrix,
                                                 join = join)
end

function weighted_average_measures(measure_matrix, confusion_matrix; join = dot)
    [join(vec(confusion_matrix[i, :] / sum(confusion_matrix[i, :])),
          vec(measure_matrix[i, :]))
     for i = 1:size(confusion_matrix, 1)]
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

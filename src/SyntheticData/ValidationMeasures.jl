module ValidationMeasures
export hard_label_accuracy_measure, hard_network_edge_accuracy_measure, network_enrichment_measure, train_loglikelihood_measure, test_loglikelihood_measure, whole_cov_data_measure, hard_network_edge_accuracy, hard_label_accuracy

using BaumWelchUtils
using BaumWelch
using EmissionDistributions

function hard_label_accuracy_measure (data_train, labels_train,
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

function train_loglikelihood_measure (data_train, labels_train,
                                      data_holdout, labels_holdout,
                                      true_model,
                                      found_estimate, found_model, found_ll)
    found_ll
end

function test_loglikelihood_measure (data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll)
    k = size(found_model.trans, 1)
    log_likelihood(data_holdout, k, found_model, dist_log_pdf)
end

function whole_cov_data_measure (data_train, labels_train,
                                 data_holdout, labels_holdout,
                                 true_model,
                                 found_estimate, found_model, found_ll)
    cov(data)
end

function network_enrichment_measure (data_train, labels_train,
                                     data_holdout, labels_holdout,
                                     true_model,
                                     found_estimate, found_model, found_ll)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    [network_enrichment(network_pair...)
     for network_pair = collect(zip(found_networks, true_networks))]
end

function network_enrichment (found_network, true_network, eps = 10e-8)
    true_edges = Set(map(t -> t[2], filter(t -> t[1] > eps, sorted_edges(true_network))))
    num_true_edges = length(true_edges)
    
    found_edges = map(t -> t[2], sorted_edges(found_network))[1:num_true_edges]

    truths = map(edge -> in((edge[1], edge[2]), true_edges) || in((edge[2], edge[1]), true_edges),
                 found_edges)

    found_true = sum(truths)
    random_true = num_true_edges * num_true_edges / length(found_network)

    found_true / random_true
end

function sorted_edges (network)
    (n, m) = size(network)
    weighted_edges = vec([(abs(network[i, j]), (i, j))
                          for i = 1:n, j = 1:m])

    sort(weighted_edges, by = t -> t[1], rev=true)
end

function hard_label_accuracy (gamma, true_labels)
    found_labels = gamma_to_labels(gamma)
    accuracy(found_labels, true_labels)
end

function hard_network_edge_accuracy (found_mat, true_mat, eps = 10e-8)
    found_network = abs(found_mat) .> eps
    true_network = abs(true_mat) .> eps
    accuracy(found_network, true_network)
end

function accuracy {T <: Any} (vec1 :: AbstractArray{T},
                              vec2 :: AbstractArray{T})
    sum(map((==), vec1, vec2)) / length(vec1)
end

end

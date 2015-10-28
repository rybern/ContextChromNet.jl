module ValidationMeasures
export hard_label_accuracy_measure, hard_network_edge_accuracy_measure, loglikelihood_measure, whole_cov_data_measure, hard_network_edge_accuracy, hard_label_accuracy

using BaumWelchUtils

function hard_label_accuracy_measure (data, true_labels, true_model,
                                      found_estimate, found_model, found_ll)
    hard_label_accuracy(found_estimate.gamma, true_labels)
end

function hard_network_edge_accuracy_measure (data, true_labels, true_model,
                                             found_estimate, found_model, found_ll)
    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(found_model)

    [hard_network_edge_accuracy(network_pair...)
     for network_pair = collect(zip(found_networks, true_networks))]
end

function loglikelihood_measure (data, true_labels, true_model,
                                found_estimate, found_model, found_ll)
    found_ll
end

function whole_cov_data_measure (data, true_labels, true_model,
                                 found_estimate, found_model, found_ll)
    cov(data)
end

function hard_label_accuracy (gamma, true_labels)
    found_labels = gamma_to_labels(gamma)
    accuracy(found_labels, true_labels)
end

function hard_network_edge_accuracy (found_mat, true_mat, eps = 10e-8)
    found_network = found_mat .> eps
    true_network = true_mat .> eps
    accuracy(found_network, true_network)
end

function accuracy {T <: Any} (vec1 :: AbstractArray{T},
                              vec2 :: AbstractArray{T})
    sum(map((==), vec1, vec2)) / length(vec1)
end

end

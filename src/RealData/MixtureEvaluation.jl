module MixtureEvaluation

export networks_edge_pca, model_to_mixture, primary_edges, networks_edge_vars, build_edges

using EdgeUtils
using MultivariateStats

function primary_edges (pca, labels, ix = 1)
    build_edges(pca.proj[:, ix])
end

function build_edges (weights, labels)
    sort(collect(zip(labels, weights)),
         by = t -> abs(t[2]),
         rev = true)
end

function networks_joint_edges (networks)
    edges = map(experiment_network_factor_edges, networks)
    EdgeUtils.max_by_weight(edges, true)
end

function model_to_mixture (model)
    [inv(cholfact(cov(state.dist))) for state = model.states]
end

function networks_edge_pca (networks)
    labels, weight_matrix = networks_to_weight_matrix(networks)
    trans, pca = edge_pca(weight_matrix)
    trans, labels, pca
end

function networks_edge_vars (networks)
    labels, weight_matrix = networks_to_weight_matrix(networks)
    vars = var(weight_matrix, 2)
    vars, labels
end

function networks_to_weight_matrix (networks)
    weighted_factor_edges = [experiment_network_factor_edges(network)
                             for network = networks]

    edge_label_vector = [edge[1] for edge = weighted_factor_edges[1]]
    label_to_ix = [edge_label_vector[i] => i
                   for i = 1:length(edge_label_vector)]
   
    edge_weights = Array(Float64,
                         length(edge_label_vector),
                         length(networks))

    for n_ix = 1:length(networks)
        for e_ix = 1:length(edge_label_vector) 
            edge = weighted_factor_edges[n_ix][e_ix]
            l_ix = label_to_ix[edge[1]]
            edge_weights[l_ix, n_ix] = edge[2]
        end
    end

    edge_label_vector, edge_weights
end
   
function edge_pca (mat)
    pca = fit(PCA, mat)
    transform(pca, mat), pca
end

end

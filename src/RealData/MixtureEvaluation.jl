module MixtureEvaluation

export networks_edga_pca

using EdgeUtils
using MultivariateStats

function networks_to_weigth_matrix (networks)
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
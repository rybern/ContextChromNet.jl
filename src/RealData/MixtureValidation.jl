module MixtureValidation

using MixtureEvaluation
using EdgeUtils

function load_mixture(source = "saved_outputs/old-k15-mixture.dump")
    open(deserialize, source)
end

function load_whole(source = "saved_outputs/whole-cov.dump")
    open(deserialize, source)
end

function pca_edges(networks = load_mixture())
    trans, labels, pca = networks_edge_pca(networks)
    build_edges(trans, labels)
end

function var_edges(networks = load_mixture())
    vars, labels = networks_edge_vars(networks)
    build_edges(vars, labels)
end

function whole_edges(whole_cov = load_whole())
    experiment_network_factor_edges(inv(cholfact(whole_cov)))
end

function joint_edges(networks = load_mixture())
    EdgeUtils.max_by_weight(map(EdgeUtils.experiment_network_factor_edges, networks))
end

function joint_enrichment(edges1, edges2;
                          joint_by = EdgeUtils.max_by_weight)
    joint_edges = joint_by(Array{((String, String), Float64)}[edges1, edges2])
    EdgeUtils.edge_enrichment(joint_edges)
end

function mixture_validation(networks = load_mixture())
    pca = pca_edges(networks)
    var = var_edges(networks)
    joint = joint_edges(networks)
    whole = whole_edges()

    println("Statewise enrichment")
    println(map(EdgeUtils.network_enrichment, networks))
    println("Whole enrichment")
    println(EdgeUtils.edge_enrichment(whole))
    println("Var enrichment")
    println(EdgeUtils.edge_enrichment(var))
    println("PCA enrichment")
    println(EdgeUtils.edge_enrichment(pca))
    println("Joint mixture by weight enrichment")
    println(EdgeUtils.edge_enrichment(joint))
    
    println("Joint-Var enightment")
    println(joint_enrichment(var, joint), ", ",
            EdgeUtils.edges_enrichment_overlap(var, joint))

    println("Var-Whole enrichment by weight")
    println(joint_enrichment(whole, var), ", ",
            EdgeUtils.edges_enrichment_overlap(var, whole))
    println("PCA-Whole enrichment by weight")
    println(joint_enrichment(whole, pca), ", ",
            EdgeUtils.edges_enrichment_overlap(whole, pca))

    println("Var-Whole enrichment by Order")
    println(joint_enrichment(whole, var, joint_by = EdgeUtils.max_by_position), ", ",
            EdgeUtils.edges_enrichment_overlap(var, whole))
    println("PCA-Whole enrichment by Order")
    println(joint_enrichment(whole, pca, joint_by = EdgeUtils.max_by_position), ", ",
            EdgeUtils.edges_enrichment_overlap(whole, pca))
    
end

end

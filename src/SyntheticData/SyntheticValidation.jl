module SyntheticValidation
export evaluate_measures, synth_validation, fit_glasso, fit_full_cov, fit_diagonal_cov, fit_glasso_

using StatsBase
using HMMTypes
using ValidationMeasures
using StateMatching
using BaumWelch
using SyntheticData
using EmissionDistributions
using BaumWelchUtils
using SimpleLogging
using Compat

using ArgParse

emission_dist_table = Dict("true" => Void,
                           "diagonal" =>  fit_diag_cov,
                           "glasso" =>  fit_glasso,
                           "full" =>  fit_full_cov)
validation_measure_table = Dict("label_accuracy" => hard_label_accuracy_measure,
                                "edge_accuracy" => hard_network_edge_accuracy_measure,
                                "test_ll" => test_loglikelihood_measure,
                                "enrichment_fold" => network_enrichment_fold_measure,
                                "edge_matches" => network_edge_matches_measure)

default_output_file = Void
default_n = 10000
default_p = 10
default_restarts = 5
default_gen_k = 5
default_seed = 5
default_emission_dist_ids = ["true", "diagonal", "glasso", "full"]
default_emission_dists = [emission_dist_table[id]
                          for id = default_emission_dist_ids]
default_emission_dist_pairs = [(id, emission_dist_table[id])
                               for id = default_emission_dist_ids]
default_validation_measure_ids = ["label_accuracy", "edge_accuracy", "test_ll", "enrichment_fold", "edge_matches"]
default_validation_measures = [validation_measure_table[id]
                               for id = default_validation_measure_ids]
default_validation_measure_pairs = Tuple{ASCIIString, Function}[(id, validation_measure_table[id])
                                                                for id = default_validation_measure_ids]
default_densities = [1.0,0.8,0.6,0.4,0.3,0.2,0.1]
default_iterations = 10
default_test_verbose = true
default_model_verbose = false

type SynthDataset
    train_data :: Array{Float64, 2}
    train_labels :: Array{Int64, 1}
    test_data :: Array{Float64, 2}
    test_labels :: Array{Int64, 1}
    true_model :: HMMStateModel
end

function toy()
    model = [("Default BaumWelch", baum_welch)]
    val = [("Network Enrichment", network_enrichment_measure),
           ("Edge Accuracy", hard_network_edge_accuracy_measure)]
    evaluate_measures(val, model, iterations = 3, verbose = 0)
end

function toy2()
    synth_validation(n = 1000, p = 5)#, emission_dists = [("True model", Void)])
end

# Bloated function
# Translate lists of interesting parameters to lists of generators/models/measures
# Also supports file output
function synth_validation(output_file :: Union{Type{Void}, AbstractString} = default_output_file;
                          n :: Int = default_n,
                          p :: Int = default_p,
                          model_restarts :: Int = default_restarts,
                          emission_dists = default_emission_dist_pairs,
                          gen_k :: Int = default_gen_k,
                          model_k :: Int = gen_k,
                          densities :: Array{Float64} = default_densities,
                          iterations :: Int = default_iterations,
                          validation_measures :: Array{Tuple{ASCIIString, Function}} = default_validation_measure_pairs,
                          test_verbose_flag :: Bool = default_test_verbose,
                          model_verbose_flag :: Bool = default_model_verbose,
                          seed :: Int = default_seed)
    test_verbose = test_verbose_flag ? 0 : Void
    model_verbose = model_verbose_flag ? 0 : Void

    logstrln("%% Starting evaluation", test_verbose)

    # Check to make sure the file is writable - fail early!
    if output_file != Void
        open(identity, output_file, "w")
    end

    # Build models from various emission distributions
    models = [(emission_label,
               emission_dist == Void ? Void :
               (data) -> baum_welch(model_restarts, data, model_k, emission_dist, verbose = model_verbose))
              for (emission_label, emission_dist) = emission_dists]

    data_generators = Tuple{ASCIIString, Function}[("Generating density $(round(density, 1))",
                                                    n -> rand_HMM_data(n, p,
                                                                       rand_HMM_model(p,
                                                                                      gen_k,
                                                                                      density = density)))
                                                   for density = densities]

    (vars, ticks, res) = evaluate_measures(validation_measures,
                                           models,
                                           data_generators,
                                           n,
                                           n,
                                           iterations = iterations,
                                           verbose = test_verbose,
                                           seed = seed)

    results = (vars, ticks, res, (n, p, gen_k, model_k, seed))

    # Dump the results
    if output_file != Void
        open(s -> serialize(s, results), output_file, "w")
    end

    logstrln("%% Complete", test_verbose)

    results
end

# Run multiple evaluation measures against multiple data generation methods and model optimizers.
# The output is in the form (variable_labels, variables_ticks, results):
#   variable_labels is an array of strings indicating the order of the indexing for the results
#   variable_ticks is an array of the names of the values for each variable, in the same order as above.
#            These names are passed in as an association list.
#   results is a high dimensional array of the measure results indexed as indicated by variable_labels.
function evaluate_measures(validation_measures :: Array{Tuple{ASCIIString, Function}},
                           # Array{Function} <: Array{Union{Function, Type{Void}}} is false.
                           # Immature language, have to give up on type system sometimes :(
                           model_optimizers, #:: Array{Tuple{ASCIIString,Union{Type{Void}, Function}}}
                           data_generators :: Array{Tuple{ASCIIString, Function}} = [("Random_HMM_p6k3", num -> rand_HMM_data(num, 6, 3))],
                           train_n = 10000,
                           holdout_n = 10000;
                           iterations :: Int64 = 1,
                           verbose = Void,
                           seed = seed)
    # Indicate the index order for the result tensor
    variable_labels  = ("model_optimizer",  "data_generator",  "validation_measure",   "iteration")
    variable_indices = ( model_optimizers,   data_generators,   validation_measures,  1:iterations)
    variable_lengths = [length(var) for var = variable_indices]


    # Ordered array of arrays of function labels,
    # e.g. [["val_measure1", "val_measure2"],
    #       ["model_opt1", "model_opt2"]]
    variable_ticks = [[t[1] for t = arr] for arr = variable_indices]

    # Array indexed by each variable tick
    # Combine all of the measures into one, which will be split up after use
    joint_measure = join_measures([measure for (name, measure) = validation_measures])

    # Seed consistently so that results are reproducible (assuming the same data generating settings)
    srand(seed)

    # Evaluate the measures with each combination of variable
    result_tensor = Array(Any, variable_lengths...)
    for (gen_ix, (gen_tick, gen_fn)) = enumerate(data_generators)
        logstrln("%% Generator $gen_ix/$(length(data_generators)) \"$gen_tick\"", verbose)
        for iter_ix = 1:iterations
            logstrln("%% Iteration $iter_ix/$iterations", verbose)
            # build a SynthDatabase with gen_fn
            all_data, all_labels, true_model = gen_fn(train_n + holdout_n)
            train_data = all_data[:, 1:train_n]
            test_data = all_data[:, train_n+1:end]
            train_labels = all_labels[1:train_n]
            test_labels = all_labels[train_n+1:end]
            dataset = SynthDataset(train_data,
                                   train_labels,
                                   test_data,
                                   test_labels,
                                   true_model)

            for (model_ix, (model_tick, model_fn)) = enumerate(model_optimizers)
                logstrln("%% Model optimizer $model_ix/$(length(model_optimizers)) \"$model_tick\"", verbose)

                measure_results = evaluate_measure(joint_measure,
                                                   model_fn,
                                                   dataset)

                for val_ix = 1:length(validation_measures)
                    result_tensor[model_ix, gen_ix, val_ix, iter_ix] = measure_results[val_ix]
                end
            end
        end
    end

    (variable_labels,
     variable_ticks,
     result_tensor)
end

# Create data using the data generator, optimize the model on the data, and then measure the model.
function evaluate_measure(# See ValidationMeasures.jl
                          validation_measure :: Function,
                          #(data) -> (estimate, model, log-likelihood)
                          model_optimizer :: Union{Type{Void},Function},
                          dataset :: SynthDataset)
    # Build model
    if model_optimizer != Void
        # Run optimizer
        (found_estimate, found_model, found_ll) = model_optimizer(dataset.train_data)



        #(found_estimate, found_model, label_confusion_matrix) = match_states(found_estimate_unordered,
                                                                             #dataset.train_labels,
                                                                             #found_model_unordered,
                                                                             #dataset.true_model)

        found_labels = gamma_to_labels(found_estimate.gamma)
        found_k = size(found_model.trans, 1)

        train_labels = dataset.train_labels
        gen_k = size(dataset.true_model.trans, 1)

        confusion_matrix = label_confusion_matrix(found_labels, found_k,
                                                  train_labels, gen_k)
    else
        # If no optimizer provided, use the truth model
        gen_k = size(dataset.true_model.trans, 1)
        found_estimate = HMMEstimate(labels_to_gamma(dataset.train_labels, gen_k))
        found_model = dataset.true_model
        found_ll = log_likelihood(dataset.train_data, gen_k, dataset.true_model, dist_log_pdf)

        state_counts = counts(dataset.train_labels, 1:gen_k)
        confusion_matrix = diagm(state_counts)
    end

    # Measure the quality of the model
    validation_measure(dataset.train_data, dataset.train_labels,
                       dataset.test_data, dataset.test_labels,
                       dataset.true_model,
                       found_estimate, found_model, found_ll,
                       confusion_matrix)
end

function best_networks(;
                       n = 10000,
                       p = 15,
                       k = 5,
                       emission_dist = fit_full_cov,
                       density = 1.0,
                       seed = Void)
    if(seed != Void)
        srand(seed)
    end

    true_model = rand_HMM_model(p, k, density = density, mean_range = 0)
    (data, true_labels) = rand_HMM_data(n, p, true_model)[1:2]

    gamma = labels_to_gamma(true_labels, k)
    new_states = emission_dist(data, gamma)
    found_networks = states_to_networks(new_states)

    true_networks = model_to_networks(true_model)
    enrichments = [ValidationMeasures.network_enrichment(t[1], t[2])
                   for t = zip(found_networks, true_networks)]

    Dict(:true_model => true_model,
         :true_labels => true_labels,
         :true_networks => true_networks,
         :found_networks => found_networks,
         :data => data,
         :enrichments => enrichments)
end

function compare_best_networks(emission_dists::Tuple{Function,Function};
                               compare_diff_by = mean,
                               k = 5,
                               kwargs...)
    seed = rand(UInt32)
    nets = [best_networks(emission_dist = emission_dist, k = k, seed = seed; kwargs...)[:found_networks]
            for emission_dist = emission_dists]
    net_pairs = collect(zip(nets[1], nets[2]))
end

function compare_best_networks_diff(args...;
                                    compare_diff_by = mean,
                                    k = 5,
                                    kwargs...)
    nets = compare_best_networks(args..., k = k; kwargs...)
    Float64[compare_diff_by(net2 - net1) for (net1, net2) = nets]
end

function compare_best_networks_diff_moments(num_repeats,
                                            args...;
                                            kwargs...)
    comparisons = Float64[mean(SyntheticValidation.compare_best_networks_diff(args...; kwargs...))
                          for i = 1:num_repeats]
    mean_and_std(comparisons)
end

function synth_data_model(;
                          n = 1000,
                          p = 15,
                          k = 5,
                          emission_dist = fit_full_cov,
                          density = 1.0,
                          match_states = true,
                          seed = Void,
                          verbose = false)
    if(seed != Void)
        srand(seed)
    end

    true_model = rand_HMM_model(p, k, density = density, mean_range = 0)
    (data, true_labels) = rand_HMM_data(n, p, true_model)[1:2]
    (unordered_estimate, unordered_model, ll) = baum_welch(data, k, emission_dist,
                                                           verbose = verbose ? 0 : Void)
    (estimate, model) = StateMatching.match_states(unordered_estimate,
                                                   unordered_model,
                                                   true_labels,
                                                   true_model)

    true_networks = model_to_networks(true_model)
    found_networks = model_to_networks(model)
    enrichments = [ValidationMeasures.network_enrichment_fold(t[1], t[2]) for t = zip(found_networks, true_networks)]

    label_accuracy = hard_label_accuracy(estimate.gamma, true_labels)

    Dict(:true_model => true_model,
         :true_labels => true_labels,
         :true_networks => true_networks,
         :found_networks => found_networks,
         :found_gamma => estimate.gamma,
         :found_labels => gamma_to_labels(estimate.gamma),
         :found_model => model,
         :found_ll => ll,
         :data => data,
         :label_accuracy => label_accuracy,
         :enrichments => enrichments)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--output-file", "-o"
            help = "The relative or full path to a Julia serial dump file."
            required = true
        "--num-observations", "-n"
            help = "Length of the training and test synthetic datasets"
            arg_type = Int
            default = default_n
        "--num-tracks", "-p"
            help = "Number of tracks in the synthetic datasets"
            arg_type = Int
            default = default_p
        "--num-gen-states", "-k"
            help = "Number of states in the true models"
            arg_type = Int
            default = default_gen_k
        "--num-model-states"
            help = "Number of states in the fit models"
            arg_type = Int
            default = default_gen_k
        "--num-model-restarts"
            help = "Number of random starting points to maximize each Baum-Welch over"
            arg_type = Int
            default = default_restarts
        "--seed"
            help = "Seed the random number generator for generating data"
            arg_type = Int
            default = default_seed
        "--iterations", "-i"
            help = "Number of tracks in the synthetic datasets"
            arg_type = Int
            default = default_iterations
        "--emission-dists"
            help = "List (comma separated, no spaces) the emission distributions to optimize models with.
Possibilities are: \"true\" (true model), \"diagonal\" (diagonal covariance matrix), \"glasso\" (graphical LASSO), \"full\" (unconstrained, full covariance matrix)"

            default = join(default_emission_dist_ids, ",")
        "--validation-measures"
            help = "List (comma separated, no spaces) the validation measures to test models with. Possibilities: \"label_accuracy\" (hard label accuracy), \"edge_accuracy\" (hard edge accuracy), \"test_ll\" (test log-likelihood), \"enrichment_fold\" (network edge enrichment), \"edge_matches\" (boolean edge matches sorted by magnitude)"
            default = join(default_validation_measure_ids, ",")
        "--densities"
            help = "List (comma separated, no spaces) the off-diagonal inverse covariance matrix densities at which to generate the synthetic data.\n"
            default = join(default_densities, ",")
        "--test-verbose"
            help = "Output testing progress to stdout"
            action = :store_true
        "--model-verbose"
            help = "Output Baum Welch optimization progress to stdout"
            action = :store_true
    end

    return parse_args(s)
end

function synth_eval_from_cli()
    args = parse_commandline()

    output_file = args["output-file"]
    n = args["num-observations"]
    p = args["num-tracks"]
    model_restarts = args["num-model-restarts"]
    gen_k = args["num-gen-states"]
    model_k = args["num-model-states"]
    model_verbose = args["model-verbose"]
    test_verbose = args["test-verbose"]
    iterations = args["iterations"]
    emission_dist_ids = split(args["emission-dists"], ",")
    emission_dists = [(id, emission_dist_table[id])
                      for id = emission_dist_ids]
    validation_measure_ids = split(args["validation-measures"], ",")
    validation_measures = Tuple{ASCIIString,Function}[(id, validation_measure_table[id])
                                                      for id = validation_measure_ids]
    densities = map(float, split(args["densities"], ","))

    synth_validation(output_file,
                     n = n,
                     p = p,
                     model_restarts = model_restarts,
                     gen_k = gen_k,
                     model_k = model_k,
                     emission_dists = emission_dists,
                     validation_measures = validation_measures,
                     densities = densities,
                     iterations = iterations,
                     test_verbose_flag = test_verbose,
                     model_verbose_flag = model_verbose)
end

if !isinteractive()
    synth_eval_from_cli()
end

end

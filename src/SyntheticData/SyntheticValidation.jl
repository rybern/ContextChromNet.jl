module SyntheticValidation
export evaluate_measures, synth_validation

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

emission_dist_table = Dict("true" => ("True model", Void),
                           "diagonal" => ("Diagonal Covariance BW", fit_diag_cov),
                           "glasso" => ("GLASSO BW", fit_glasso),
                           "full" => ("Full Covariance BW", fit_full_cov))
validation_measure_table = Dict("label_accuracy" => ("Label Accuracy", hard_label_accuracy_measure),
                                "edge_accuracy" => ("Edge Accuracy",hard_network_edge_accuracy_measure),
                                "test_ll" => ("Test Log-Likelihood", test_loglikelihood_measure),
                                "edge_enrichment" => ("Network Enrichment", network_enrichment_measure))

default_output_file = Void
default_n = 10000
default_p = 10
default_k = 5
default_emission_dist_ids = ["true", "diagonal", "glasso", "full"]
default_emission_dists = [emission_dist_table[id]
                          for id = default_emission_dist_ids]
default_validation_measure_ids = ["label_accuracy", "edge_accuracy", "test_ll", "edge_enrichment"]
default_validation_measures = [validation_measure_table[id]
                               for id = default_validation_measure_ids]
default_densities = [1.0,0.8,0.6,0.4,0.3,0.2,0.1]
default_iterations = 10
default_test_verbose = true
default_model_verbose = false

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
function synth_validation(output_file :: AbstractString = default_output_file;
                              n :: Int = default_n,
                              p :: Int = default_p,
                              k :: Int = default_k,
                              emission_dists = default_emission_dists,
                              validation_measures :: Array{Tuple{ASCIIString, Function}} = default_validation_measures,
                              densities :: Array{Float64} = default_densities,
                              iterations :: Int = default_iterations,
                              test_verbose_flag :: Bool = default_test_verbose,
                              model_verbose_flag :: Bool = default_model_verbose)
    test_verbose = test_verbose_flag ? 0 : Void
    model_verbose = model_verbose_flag ? 0 : Void

    logstrln("Starting evaluation", test_verbose)

    # Check to make sure the file is writable - fail early!
    if output_file != Void
        open(identity, output_file, "w")
    end

    # Build models from various emission distributions
    models = [(emission_label,
               emission_dist == Void ? Void :
               (data, k) -> baum_welch(5, data, k, emission_dist, verbose = model_verbose))
              for (emission_label, emission_dist) = emission_dists]

    # Build data generators from various sparsities
    logstrln("Pregenerating models", test_verbose)

    sparsities = 1 - densities
    data_models = [rand_HMM_model(p, k, sparsity = sparsity)
                   for sparsity = sparsities]
    data_generators = Tuple{ASCIIString, Function}[("Generating density $(round(1-sparsity, 1))",
                                                    n -> rand_HMM_data(n, p, rand_HMM_model(p, k, sparsity = sparsity)))
                                                   for sparsity = sparsities]

    # Run all of the evaluations
    results = evaluate_measures(validation_measures,
                                models,
                                data_generators,
                                n,
                                n,
                                iterations = iterations,
                                verbose = test_verbose)

    # Dump the results
    if output_file != Void
        open(s -> serialize(s, results), output_file, "w")
    end

    logstrln("Complete", test_verbose)

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
                           args...;
                           iterations :: Int64 = 1,
                           verbose = Void)
    # Indicate the index order for the result tensor
    variable_labels  = ("model_optimizer",  "data_generator",  "validation_measure",   "iteration")
    variable_indices = ( model_optimizers,   data_generators,   validation_measures,  1:iterations)
    variable_lengths = [length(var) for var = variable_indices]

    # Ordered array of arrays of function labels,
    # e.g. [["val_measure1", "val_measure2"],
    #       ["model_opt1", "model_opt2"]]
    variable_ticks = [[t[1] for t = arr] for arr = variable_indices]

    # Array indexed by each variable tick
    result_tensor = convert(Array{Any, 4}, zeros(variable_lengths...))

    # Combine all of the measures into one, which will be split up after use
    joint_measure = join_measures([measure for (name, measure) = validation_measures])

    # Evaluate the measures with each combination of variable
    for (model_ix, (model_tick, model_fn)) = enumerate(model_optimizers)
        logstrln("Model optimizer $model_ix/$(length(model_optimizers)) \"$model_tick\"", verbose)
        for (gen_ix, (gen_tick, gen_fn)) = enumerate(data_generators)
            logstrln("Generator $gen_ix/$(length(data_generators)) \"$gen_tick\"", verbose)
            for iter_ix = 1:iterations
                #logstrln("\tIteration $iter_ix/$iterations", verbose)

                measure_results = evaluate_measure(joint_measure,
                                                   model_fn,
                                                   gen_fn,
                                                   args...)

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
                          #(data, k) -> (estimate, model, log-likelihood)
                          model_optimizer :: Union{Type{Void},Function},
                          # n -> (data, labels, model)
                          data_generator :: Function = num -> rand_HMM_data(num, 6, 3),
                          train_n = 10000,
                          holdout_n = 10000)
    # Initialize data
    data_all, labels_all, true_model = data_generator(train_n + holdout_n)
    data_train = data_all[:, 1:train_n]
    data_holdout = data_all[:, train_n+1:end]
    labels_train = labels_all[1:train_n]
    labels_holdout = labels_all[train_n+1:end]

    k = size(true_model.trans, 1)

    # Build model
    if model_optimizer != Void
        # Run optimizer
        (found_estimate_unordered, found_model_unordered, found_ll) =
            model_optimizer(data_train, k)

        (found_estimate, found_model) = match_states(found_estimate_unordered,
                                                     found_model_unordered,
                                                     labels_train,
                                                     true_model)
    else
        # If no optimizer provided, use the truth model
        found_estimate = HMMEstimate(labels_to_gamma(labels_train, k))
        found_model = true_model
        found_ll = log_likelihood(data_train, k, true_model, dist_log_pdf)
    end

    # Measure the quality of the model
    validation_measure(data_train, labels_train,
                       data_holdout, labels_holdout,
                       true_model,
                       found_estimate, found_model, found_ll)
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
        "--num-states", "-k"
            help = "Number of states in the true and fit models"
            arg_type = Int
            default = default_k
        "--iterations", "-i"
            help = "Number of tracks in the synthetic datasets"
            arg_type = Int
            default = default_iterations
        "--emission-dists"
            help = "List (comma separated, no spaces) the emission distributions to optimize models with.
Possibilities are: \"true\" (true model), \"diagonal\" (diagonal covariance matrix), \"glasso\" (graphical LASSO), \"full\" (unconstrained, full covariance matrix)"

            default = join(default_emission_dist_ids, ",")
        "--validation-measures"
            help = "List (comma separated, no spaces) the validation measures to test models with. Possibilities: \"label_accuracy\" (hard label accuracy), \"edge_accuracy\" (hard edge accuracy), \"test_ll\" (test log-likelihood), \"edge_enrichment\" (network edge enrichment)"
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
    k = args["num-states"]
    model_verbose = args["model-verbose"]
    test_verbose = args["test-verbose"]
    iterations = args["iterations"]
    emission_dist_ids = split(args["emission-dists"], ",")
    emission_dists = [emission_dist_table[id]
                      for id = emission_dist_ids]
    validation_measure_ids = split(args["validation-measures"], ",")
    validation_measures = Tuple{ASCIIString,Function}[validation_measure_table[id]
                                                      for id = validation_measure_ids]
    densities = map(float, split(args["densities"], ","))

    synth_validation(output_file,
                         n = n,
                         p = p,
                         k = k,
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

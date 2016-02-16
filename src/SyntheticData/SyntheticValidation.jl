module SyntheticValidation
export evaluate_measures, run_synth_validation

using HMMTypes
using ValidationMeasures
using StateMatching
using BaumWelch
using SyntheticData
using EmissionDistributions
using BaumWelchUtils
using SimpleLogging
using Compat

function toy()
    model = [("Default BaumWelch", baum_welch)]
    val = [("Network Enrichment", network_enrichment_measure),
           ("Edge Accuracy", hard_network_edge_accuracy_measure)]
    evaluate_measures(val, model, iterations = 3, verbose = 0)
end

function toy2()
    run_synth_validation(emission_fitters = [fit_diag_cov, fit_full_cov],
                         validations = [hard_label_accuracy_measure,
                                        hard_network_edge_accuracy_measure,
                                        test_loglikelihood_measure,
                                        train_loglikelihood_measure,
                                        network_enrichment_measure],
                         sparsities = [.5],
                         n = 1000,
                         p = 30,
                         repeat = Void,
                         model_verbose = false,
                         eval_verbose = true)
end

function toy3()
    run_synth_validation("saved_outputs/synthetic:gen-model-measure-iter.dump",
                         emission_fitters = [fit_full_cov, Void],
                         validations = [hard_label_accuracy_measure,
                                        hard_network_edge_accuracy_measure,
                                        test_loglikelihood_measure,
                                        train_loglikelihood_measure,
                                        network_enrichment_measure],
                         sparsities = [0],
                         repeat = 1,
                         model_verbose = false)
end

function run_synth_validation(output_file = Void;
                              n = 100000,
                              p = 30,
                              k = 5,
                              emission_dists = [Void,
                                                  fit_diag_cov,
                                                  fit_glasso,
                                                  fit_full_cov],
                              validations = [hard_label_accuracy_measure,
                                             hard_network_edge_accuracy_measure,
                                             test_loglikelihood_measure,
                                             train_loglikelihood_measure,
                                             network_enrichment_measure],
                              sparsities = [0,
                                            0.5,
                                            0.7,
                                            0.8,
                                            0.9],
                              repeat = 10,
                              eval_verbose = true,
                              model_verbose = false)
    verbose = eval_verbose ? 0 : Void

    logstrln("Starting evaluation", verbose)

    # Check to make sure the file is writable - fail early!
    if output_file != Void
        open(identity, output_file, "w")
    end

    models = [emission_dist == Void ? Void :
              (data, k) -> baum_welch(5, data, k, emission_dist, verbose)
              for emission_dist = emission_dists]

    logstrln("Pregenerating models", verbose)

    data_models = [rand_HMM_model(p, k, sparsity = sparsity)
                   for sparsity = sparsities]
    data_generators = [(n -> rand_HMM_data(n, p, data_model))
                       for data_model = data_models]

    results = evaluate_measures(validations,
                                models,
                                data_generators,
                                n,
                                n,
                                repeat = repeat,
                                verbose = verbose)


    if output_file != Void
        open(s -> serialize(s, results), output_file, "w")
    end

    logstrln("Complete", verbose)

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
                           verbose = Void,
                           kwargs...)
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
        logstrln("Model optimizer $model_ix/$(length(model_optimizers))", verbose)
        for (gen_ix, (gen_tick, gen_fn)) = enumerate(data_generators)
            logstrln("Generator $gen_ix/$(length(data_generators))", verbose)
            for iter_ix = 1:iterations
                logstrln("\tIteration $iter_ix/$iterations", verbose)

                measure_results = evaluate_measure(joint_measure,
                                                   model_fn,
                                                   gen_fn,
                                                   args...;
                                                   kwargs...)

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
                          data_generator :: Function = num -> rand_HMM_data(num, 6, 3);
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

end

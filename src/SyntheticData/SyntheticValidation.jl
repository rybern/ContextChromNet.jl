module SyntheticValidation
export evaluate_measures, run_synth_validation

using HMMTypes
using ValidationMeasures
using StateMatching
using BaumWelch
using SyntheticData
using EmissionDistributions
using BaumWelchUtils
using Logging

function ll_test()
    train_n = 10000
    holdout_n = 10000

    data_all, labels_all, true_model = rand_HMM_data(train_n + holdout_n,
                                                     5,
                                                     5)
    data_train = data_all[:, 1:train_n]
    data_holdout = data_all[:, train_n+1:end]
    labels_train = labels_all[1:train_n]
    labels_holdout = labels_all[train_n+1:end]



    estimate, found_model, train_returned_ll = baum_welch(5, data_train, 5)

    train_true_ll = log_likelihood(data_train, 5, true_model, dist_log_pdf)
    train_found_ll = log_likelihood(data_train, 5, found_model, dist_log_pdf)
    holdout_true_ll = log_likelihood(data_holdout, 5, true_model, dist_log_pdf)
    holdout_found_ll = log_likelihood(data_holdout, 5, found_model, dist_log_pdf)

    train_returned_ll, train_found_ll, train_true_ll, holdout_found_ll, holdout_true_ll
end

function toy()
    model = [baum_welch]
    val = [network_enrichment_measure,
           hard_network_edge_accuracy_measure]
    evaluate_measures(val, model, repeat = 3)
end

function toy2()
    run_synth_validation(emission_fitters = [fit_diag_cov, fit_full_cov, Nothing],
                         validations = [hard_label_accuracy_measure,
                                        hard_network_edge_accuracy_measure,
                                        test_loglikelihood_measure,
                                        train_loglikelihood_measure,
                                        network_enrichment_measure],
                         sparsities = [.5],
                         n = 1000,
                         p = 30,
                         repeat = Nothing,
                         model_verbose = false,
                         eval_verbose = true)
end

function toy3()
    run_synth_validation("saved_outputs/synthetic:gen-model-measure-iter.dump",
                         emission_fitters = [fit_full_cov, Nothing],
                         validations = [hard_label_accuracy_measure,
                                        hard_network_edge_accuracy_measure,
                                        test_loglikelihood_measure,
                                        train_loglikelihood_measure,
                                        network_enrichment_measure],
                         sparsities = [0],
                         repeat = 1,
                         model_verbose = false)
end

function run_synth_validation(output_file = Nothing;
                              n = 100000,
                              p = 30,
                              k = 5,
                              emission_fitters = [Nothing,
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
    if eval_verbose
        logstrln("Starting evaluation")
    end

    if output_file != Nothing
        open(identity, output_file, "w")
    end    

    models = [emission_dist == Nothing ? Nothing :
              (data, k) -> baum_welch(5, data, k, emission_dist,
                                      verbose = model_verbose ? eval_verbose : Nothing)
              for emission_dist = emission_fitters]

    if (eval_verbose)
        logstrln("Pregenerating models")
    end

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
                                verbose = eval_verbose)

    if output_file != Nothing
        open(s -> serialize(s, results), output_file, "w")
    end

    if (eval_verbose)
        logstrln("Complete")
    end

    results
end

function evaluate_measures (validation_measure :: Function,
                            model_optimizer :: Union(Type{Nothing},Function),
                            args...;
                            repeat :: Integer = 10,
                            verbose = true,
                            kwargs...)
    function repeat_evaluate(i)
        if verbose 
            logstrln("\tIteration $i/$repeat")
        end
        
        evaluate_measures(args...;
                          repeat = Nothing,
                          verbose = verbose,
                          kwargs...)
    end

    [repeat_evaluate(iteration)
     for iteration = 1:repeat]
end


# if repeat != Nothing:
#     results[iteration_ix]
# else:
#     results
# end
function evaluate_measures (# data, true_lables,
                            # true_model, found_estimate, found_model,
                            # found_ll -> X
                            validation_measure :: Function,
                            #(data, k) -> (estimate, model, log-likelihood)
                            model_optimizer :: Union(Type{Nothing},Function),
                            # n -> (data, labels, model)
                            data_generator :: Function = num -> rand_HMM_data(num, 6, 3),
                            train_n = 10000,
                            holdout_n = 10000;
                            repeat = Nothing, # Should get here if integer
                            verbose = true)
    if typeof(repeat) <: Integer
        function repeat_evaluate(i)
            if verbose 
                logstrln("\tIteration $i/$repeat")
            end
            
            evaluate_measures(validation_measure,
                              model_optimizer,
                              data_generator,
                              train_n,
                              holdout_n;
                              repeat = Nothing,
                              verbose = verbose)
        end

        return [repeat_evaluate(iteration)
                for iteration = 1:repeat]
    end

    data_all, labels_all, true_model = data_generator(train_n + holdout_n)
    data_train = data_all[:, 1:train_n]
    data_holdout = data_all[:, train_n+1:end]
    labels_train = labels_all[1:train_n]
    labels_holdout = labels_all[train_n+1:end]


    k = size(true_model.trans, 1)
    if model_optimizer != Nothing
        (found_estimate_unordered, found_model_unordered, found_ll) =
            model_optimizer(data_train, k)
    
        (found_estimate, found_model) = match_states(found_estimate_unordered,
                                                     found_model_unordered,
                                                     labels_train,
                                                     true_model)
    else
        found_estimate = HMMEstimate(labels_to_gamma(labels_train, k))
        found_model = true_model
        found_ll = log_likelihood(data_train, k, true_model, dist_log_pdf)
    end

    validation_measure(data_train, labels_train,
                       data_holdout, labels_holdout,
                       true_model,
                       found_estimate, found_model, found_ll)
end

# if repeat != Nothing:
#     results[measure_ix][iteration_ix]
# else:
#     results[measure_ix]
# end
function evaluate_measures(validation_measures :: Array{Function},
                           model_optimizer :: Union(Type{Nothing},Function),
                           args...;
                           repeat = Nothing,
                           kwargs...)
    function validation_measure (data_train, labels_train,
                                 data_holdout, labels_holdout,
                                 true_model,
                                 found_estimate, found_model, found_ll)
        [measure(data_train, labels_train,
                 data_holdout, labels_holdout,
                 true_model,
                 found_estimate, found_model, found_ll)
         for measure = validation_measures]
    end

    results = evaluate_measures(validation_measure,
                                model_optimizer,
                                args...;
                                repeat = repeat)

    if repeat != Nothing
        [[iteration[measure_ix] for iteration = results]
         for measure_ix = 1:length(validation_measures)]
    else
        results
    end
end

# if repeat != Nothing:
#     results[model_ix][measure_ix][iteration_ix]
# else:
#     results[model_ix][measure_ix]
# end
function evaluate_measures(validation_measures :: Array{Function},
                           model_optimizers :: Array{Union(Type{Nothing},Function)},
                           args...;
                           verbose = true,
                           include_true = true,
                           kwargs...)
    seed = 11

    function evaluate_optimizer (optimizer_ix)
        if verbose
            logstrln("Model optimizer $optimizer_ix/$(length(model_optimizers))")
        end

        # each optimizer should get the same data
        srand(seed)

        evaluate_measures(validation_measures,
                          model_optimizers[optimizer_ix],
                          args...;
                          kwargs...)
    end

    results = [evaluate_optimizer(ix)
               for ix = 1:length(model_optimizers)]



    results
end

# if repeat != Nothing:
#     results[gen_ix][model_ix][measure_ix][iteration_ix]
# else:
#     results[gen_ix][model_ix][measure_ix]
# end
function evaluate_measures(validation_measures :: Array{Function},
                           model_optimizers :: Array{Union(Type{Nothing},Function)},
                           data_generators :: Array{Function},
                           args...;
                           verbose = true,
                           kwargs...)
    function evaluate_generator (data_generator_ix)
        if verbose
            logstrln("Generator $data_generator_ix/$(length(data_generators))")
        end

        evaluate_measures(validation_measures,
                          model_optimizers,
                          data_generators[data_generator_ix],
                          args...;
                          kwargs...)  
    end 

    [evaluate_generator(ix)
     for ix = 1:length(data_generators)]                     
end

end


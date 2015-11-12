module SyntheticFigures
export run_standard_figures, measure_figure

using BarchartPlots

gen_names = map(string, [1,
                         0.5,
                         0.3,
                         0.2,
                         0.1])
model_names = ["Truth",
               "Diag Cov",
               "GLasso",
               "Full Cov"]
measure_names = ["Label accuracy",
                 "Network accuracy",
                 "Negative Test Log-Likelihood",
                 "Train Log-Likelihood Accuracy",
                 "Network enrichment fold"]

measure_perm = [1:5]
model_perm = [2, 3, 4, 1]
gen_perm = reverse([1:5])

function run_standard_figures(results_file)
    results = open(deserialize, results_file)

    # label accuracy
    measure_figure(results, 1, "Label Accuracy versus Sparsity", Nothing,
                   y_limits = true,
                   save_file = "results/label-acc.png")

    # network acc
#    measure_figure(results,
#                   2,
#                   "Average network accuracy versus Sparsity",
#                   Nothing,
#                   preprocess = mean,
#                   save_file = "results/network_acc.png")

    # loglikelihood
    measure_figure(results, 3, "Negative Test Log-Likelihood versus Sparsity", Nothing,
                   preprocess = (-),
                   save_file = "results/neg-test-loglike.png",
                   y_limits = true)

    # network acc
    measure_figure(results,
                   5,
                   "Average network enrichment fold versus Sparsity",
                   Nothing,
                   preprocess = mean,
                   save_file = "results/net-enrichment.png",
                   model_ixs = [3, 4, 1])
end

function measure_figure(results,
                        measure_ix,
                        figure_title,
                        relative = Nothing;
                        model_ixs = model_perm,
                        gen_ixs = gen_perm,
                        preprocess = identity,
                        save_file = Nothing,
                        y_limits = false)

    println(model_ixs)

    means, stds = measure_moments(results,
                                  measure_ix,
                                  relative,
                                  preprocess = preprocess)

    n_gens = length(means)
    n_models = length(means[1])

    bars = [[Array{Float64, 1}[[means[gen_ix][model_ix]
                                for gen_ix = gen_ixs],
                               [stds[gen_ix][model_ix]
                                for gen_ix = gen_ixs]],
             model_names[model_ix]]
            for model_ix = model_ixs]

    println(bars)

    grouped_bar_plot(bars,
                     gen_names[gen_perm],
                     "Density",
                     measure_names[measure_perm][measure_ix],
                     figure_title,
                     dumpfile = save_file == Nothing ? false : save_file,
                     bar_width = .15,
                     y_limits = y_limits)
end

function split_result_measures (res)
    gen_len = length(res)
    model_len = length(res[1])
    measure_len = length(res[1][1])
 
    [[[res[gen_ix][model_ix][measure_ix]
       for model_ix = 1:model_len]
      for gen_ix = 1:gen_len]
     for measure_ix = 1:measure_len]
end

function measure_moments (results,
                          measure_index,
                          relative = Nothing;
                          preprocess = identity)
    res = split_result_measures(results)[measure_index]
    gen_len = length(res)
    model_len = length(res[1])

    res = [[map(preprocess, res[gen_ix][model_ix])
             for model_ix = 1:model_len]
            for gen_ix = 1:gen_len]

    if relative != Nothing
        res = scale_measure(res, measure_index, relative)
    end

    stds = [[std(convert(Array{Float64, 1}, res[gen_ix][model_ix]))
             for model_ix = 1:model_len]
            for gen_ix = 1:gen_len]

    means = [[mean(res[gen_ix][model_ix])
              for model_ix = 1:model_len]
             for gen_ix = 1:gen_len]

    means, stds
end

function scale_measure (measure_results,
                        measure_index,
                        reference_model = 1)
    gen_len = length(measure_results)
    model_len = length(measure_results[1])
    iteration_len = length(measure_results[1][1])

    for gen_ix = 1:gen_len
        for iteration_ix = 1:iteration_len
            scaled = [measure_results[gen_ix][model_ix][iteration_ix]
                      for model_ix = 1:model_len] / measure_results[gen_ix][reference_model][iteration_ix]
            
            for model_ix = 1:model_len
                measure_results[gen_ix][model_ix][iteration_ix] = scaled[model_ix]
            end
        end
    end

    measure_results
end

end

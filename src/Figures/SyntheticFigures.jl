module SyntheticFigures
export run_standard_figures, measure_figure

using BarchartPlots
using PyPlot.(figure)
using PyPlot.(savefig)
using SimplePlot
using HypothesisTests

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
                 "Test Log-Likelihood",
                 "Network enrichment fold"]

measure_perm = [1:5]
model_perm = [2, 3, 4, 1]
gen_perm = reverse([1:5])

function run_standard_figures(results_file = "saved_outputs/synthetic:n100k-k30-v3.dump")
    results = open(deserialize, results_file)

    label_acc = true
    network_acc = false
    tll = true
    network_enrich = true

    # label accuracy
    if label_acc
        measure_figure_(results, 1, "Label Accuracy versus Model Density", Void,
                        y_limits = true,
                        save_file = "results/label-acc.png",
                        model_ixs = model_perm[1:3])
        pairs = measure_pair_test(results, 1)
        println("Pair tests:")
        map(println, pairs);
    end

    # Network acc
    if network_acc
        measure_figure(results,
                       2,
                       "Average network accuracy versus Model Density",
                       Void,
                       preprocess = mean,
                       save_file = "results/network_acc.png")
    end

    # loglikelihood
    if tll
        measure_figure_(results, 3, "Negative Test Log-Likelihood versus Model Density", Void,
                        preprocess = (x -> -(x / 100000)),
                        save_file = "results/neg-test-loglike.png",
                        y_limits = true)
    end

    # network enrich
    if network_enrich
        measure_figure_(results,
                        5,
                        "Average network enrichment fold versus Model Density",
                        Void,
                        preprocess = mean,
                        save_file = "results/net-enrichment.png",
                        model_ixs = [3, 4, 1])
    end
end

function measure_pair_test(results,
                           measure_ix,
                           preprocess = identity)
    # gen, model, iteration
    res = split_result_measures(results)[measure_ix]

    gen_len = length(res)
    model_len = length(res[1])

    [[pvalue(SignedRankTest(float(res[gen_ix][model_perm[model_ix]]),
                            float(res[gen_ix][model_perm[model_ix + 1]])))
      for model_ix = 1:(model_len-1)]
     for gen_ix = 1:gen_len]
end  

function measure_figure_(results,
                         measure_ix,
                         figure_title,
                         relative = Void;
                         model_ixs = model_perm,
                         gen_ixs = gen_perm,
                         preprocess = identity,
                         save_file = Void,
                         y_limits = false)

    means, stds = measure_moments(results,
                                  measure_ix,
                                  relative,
                                  preprocess = preprocess)

    n_gens = length(means)
    n_models = length(means[1])

    bars = [bar(gen_names[gen_ixs],
                [means[gen_ix][model_ix] for gen_ix = 1:n_gens],
                model_names[model_ix])
            for model_ix = model_ixs]

    if y_limits
        bars_ = [[Array{Float64, 1}[[means[gen_ix][model_ix]
                                    for gen_ix = gen_ixs],
                                   [stds[gen_ix][model_ix]
                                    for gen_ix = gen_ixs]],
                 model_names[model_ix]]
                for model_ix = model_ixs]
        y = reduce(vcat, [bari[1] for bari = bars_])
        y_min = minimum(y)
        y_max = maximum(y)
        y_range = y_max - y_min
        buffer = y_range / 20;

        y_lower = y_min - buffer
        y_upper = y_max + buffer

        ylim = [y_lower, y_upper]
    else 
        ylim = false
    end

#    figure()

    if ylim == false 
        fig = plot(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_perm][measure_ix],
                   xlabel = "Density",
                   title = figure_title)
    else
        fig = plot(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_perm][measure_ix],
                   xlabel = "Density",
                   title = figure_title,
                   ylim = ylim)
    end
    


    if save_file != Void
        savefig(save_file, dpi = 200)
    end
    show()
end


function measure_figure(results,
                        measure_ix,
                        figure_title,
                        relative = Void;
                        model_ixs = model_perm,
                        gen_ixs = gen_perm,
                        preprocess = identity,
                        save_file = Void,
                        y_limits = false)

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

    grouped_bar_plot(bars,
                     gen_names[gen_perm],
                     "Density",
                     measure_names[measure_perm][measure_ix],
                     figure_title,
                     dumpfile = save_file == Void ? false : save_file,
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
                          relative = Void;
                          preprocess = identity)
    res = split_result_measures(results)[measure_index]
    gen_len = length(res)
    model_len = length(res[1])

    res = [[map(preprocess, res[gen_ix][model_ix])
             for model_ix = 1:model_len]
            for gen_ix = 1:gen_len]

    if relative != Void
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

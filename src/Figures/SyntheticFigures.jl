module SyntheticFigures
export run_standard_figures, measure_figure

#using BarchartPlots
#using PyPlot.(figure)
using PyPlot.(savefig)
using SimplePlot
using HypothesisTests
using ArgParse

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

measure_perm = collect(1:4)
model_perm = collect(1:3)
gen_perm = collect(1:6)

function run_standard_figures(results_file = "saved_outputs/synth-full-2-24-16.dump",
                              output_dir = "results/synthetic_figures")
    if(!isdir(output_dir))
        mkdir(output_dir)
    end

    (vars, ticks, results, specs) = open(deserialize, results_file)

    model_names = ticks[1]
    measure_names = ticks[3]
    # Only take the density number to avoid filling up the whole plot
    gen_names = map(a -> a[length(a)-2:length(a)], ticks[2])

    label_acc = true
    tll = true
    network_enrich = true

    figures = []

    # label accuracy
    if label_acc
        fig = measure_bar_figure_simple(results, 1, "Label Accuracy versus Model Density", Void,
                                    y_limits = true,
                                    save_file = "$output_dir/label-acc.png",
                                    model_ixs = model_perm[1:3],
                                    model_names = model_names,
                                    measure_names = measure_names,
                                    gen_names = gen_names)
        push!(figures, fig)
        pairs = measure_pair_test(results, 1)
        println("Pair tests:")
        map(println, pairs);
    end

    # log-likelihood
    if tll
        fig = measure_bar_figure_simple(results, 3, "Negative Test Log-Likelihood versus Model Density", Void,
                                    preprocess = (x -> -(x / 100000)),
                                    save_file = "$output_dir/neg-test-loglike.png",
                                    y_limits = true,
                                    model_names = model_names,
                                    measure_names = measure_names,
                                    gen_names = gen_names)
        push!(figures, fig)
    end

    # network enrichment
    if network_enrich
        fig = measure_bar_figure_simple(results,
                                    4,
                                    "Average network enrichment fold versus Model Density",
                                    Void,
                                    preprocess = mean,
                                    save_file = "$output_dir/net-enrichment.png",
                                    model_names = model_names,
                                    measure_names = measure_names,
                                    gen_names = gen_names)
        push!(figures, fig)
    end

    # network enrichment
    if enrich_curve
        fig = measure_line_figure_simple(results,
                                         5,
                                         "Enrichment Curve",
                                         Void,
                                         preprocess = mean,
                                         save_file = "$output_dir/net-enrichment-curve.png",
                                         model_names = model_names,
                                         measure_names = measure_names,
                                         gen_names = gen_names)
        push!(figures, fig)
    end

    figures
end

function measure_pair_test(results,
                           measure_ix,
                           preprocess = identity)
    # gen, model, iteration
    res = split_result_measures(results)[measure_ix]

    gen_len = length(res)
    model_len = length(res[1])

    println("$gen_len, $model_len")
    [[pvalue(SignedRankTest(convert(Array{Float64, 1}, res[gen_ix][model_ix]),
                            convert(Array{Float64, 1}, res[gen_ix][model_ix + 1])))
      for model_ix = 1:(model_len-1)]
     for gen_ix = 1:gen_len]
end  

function measure_bar_figure_simple(results,
                                   measure_ix,
                                   figure_title,
                                   relative = Void;
                                   model_ixs = model_perm,
                                   gen_ixs = gen_perm,
                                   preprocess = identity,
                                   save_file = Void,
                                   y_limits = false,
                                   model_names = model_names,
                                   measure_names = measure_names,
                                   gen_names = gen_names)

    means, stds = measure_moments(results,
                                  measure_ix,
                                  relative,
                                  preprocess = preprocess)

    n_gens = length(means)
    n_models = length(means[1])

    bars = [bar(convert(Array{ASCIIString,1}, gen_names[gen_ixs]),
                Float64[means[gen_ix][model_ix] for gen_ix = 1:n_gens],
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

#   figure()

    if ylim == false
        #println("bars here")
        #for model_ix = model_ixs
            #println("bar($(convert(Array{ASCIIString,1}, gen_names[gen_ixs])),
                #$(Float64[means[gen_ix][model_ix] for gen_ix = 1:n_gens]),
                #\"$(model_names[model_ix])\")")
        #end
        fig = axis(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_ix],
                   xlabel = "Density",
                   title = figure_title)
    else
        fig = axis(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_ix],
                   xlabel = "Density",
                   title = figure_title,
                   ylim = ylim)
    end

    if save_file != Void
        savefig(save_file, dpi = 200)
    end

    fig
end

function measure_line_figure_simple(results,
                                   measure_ix,
                                   figure_title,
                                   relative = Void;
                                   model_ixs = model_perm,
                                   gen_ixs = gen_perm,
                                   preprocess = identity,
                                   save_file = Void,
                                   y_limits = false,
                                   model_names = model_names,
                                   measure_names = measure_names,
                                   gen_names = gen_names)

    means, stds = measure_moments(results,
                                  measure_ix,
                                  relative,
                                  preprocess = preprocess)

    n_gens = length(means)
    n_models = length(means[1])

    bars = [bar(convert(Array{ASCIIString,1}, gen_names[gen_ixs]),
                Float64[means[gen_ix][model_ix] for gen_ix = 1:n_gens],
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

#   figure()

    if ylim == false
        #println("bars here")
        #for model_ix = model_ixs
            #println("bar($(convert(Array{ASCIIString,1}, gen_names[gen_ixs])),
                #$(Float64[means[gen_ix][model_ix] for gen_ix = 1:n_gens]),
                #\"$(model_names[model_ix])\")")
        #end
        fig = axis(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_ix],
                   xlabel = "Density",
                   title = figure_title)
    else
        fig = axis(bars...,
                   legend = "upper right",
                   ylabel = measure_names[measure_ix],
                   xlabel = "Density",
                   title = figure_title,
                   ylim = ylim)
    end

    if save_file != Void
        savefig(save_file, dpi = 200)
    end

    fig
end



function measure_figure_(results,
                         measure_ix,
                         figure_title,
                         relative = Void;
                         model_ixs = model_perm,
                         gen_ixs = gen_perm,
                         preprocess = identity,
                         save_file = Void,
                         y_limits = false,
                         model_names = model_names,
    measure_names = measure_names,
    gen_names = gen_names)

    println(model_names, measure_names, gen_names)
    println(size(results))

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

function split_result_measures_(res)
    gen_len = size(res, 2)
    model_len = size(res, 1)
    measure_len = size(res, 3)

    [[[res[model_ix][gen_ix][measure_ix]
       for model_ix = 1:model_len]
      for gen_ix = 1:gen_len]
     for measure_ix = 1:measure_len]
end

# for new result format
function split_result_measures(res)
    (n_models, n_gens, n_measures, n_iters) = size(res)

    [[[vec(res[model_ix, gen_ix, measure_ix, :])
       for model_ix = 1:n_models]
      for gen_ix = 1:n_gens]
     for measure_ix = 1:n_measures]
end

function measure_moments(results,
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

function scale_measure(measure_results,
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

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--output-dir", "-o"
            help = "path to directory in which to store the figures"
            arg_type = ASCIIString
            required = true
        "--results-file", "-i"
            help = "path to the Julia serialized file containing CCN synthetic test results"
            arg_type = ASCIIString
            required = true
    end

    return parse_args(s)
end

function synth_figures_from_cli()
    args = parse_commandline()

    output_dir = args["output-dir"]
    results_file = args["results-file"]

    run_standard_figures(results_file,
                         output_dir)
end

if !isinteractive()
    synth_figures_from_cli()
end

end

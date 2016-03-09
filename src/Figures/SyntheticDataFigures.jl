module SyntheticDataFigures

using PyPlot.(savefig)
using SimplePlot
using HypothesisTests
using ArgParse

function build_figures(results_file,
                       output_directory;
                       include_measures = Void)
    results = open(deserialize, results_file)
    (vars, ticks, res, specs) = results

    # make sure the results version is okay
    @assert vars == ("model_optimizer","data_generator","validation_measure","iteration")

    measures = collect(enumerate(ticks[3]))
    if include_measures != Void
        filter!(m -> in(m[2], include_measures), measures)
    end

    [plotting_functions[measure](ticks, slice(res, :, :, measure_ix, :))
     for (measure_ix, measure) = measures]
end

function truncate_generator_names(generator_names)
    [name[length(name)-2:length(name)] for name = generator_names]
end

function average_iteration_bars(ticks, measure_res; preprocess = identity)
    bars = [bar(truncate_generator_names(ticks[2]), # list of density strings
                vec(convert_mean(map(preprocess,
                                     slice(measure_res, model_ix, :, :)),
                                 2)), #average each generator over iterations
                model_name,
                color = colors[model_ix])
            for (model_ix, model_name) = enumerate(ticks[1])]
end

function plot_edge_accuracy(ticks, measure_res)
    bars = average_iteration_bars(ticks, measure_res,
                                  preprocess = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Edge Accuracy",
               xlabel = "Generating Network Density",
               title = "Edge Accuracy vs. Generating Density")
end

function plot_label_accuracy(ticks, measure_res)
    bars = average_iteration_bars(ticks, measure_res)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Label Accuracy",
               xlabel = "Generating Network Density",
               title = "Label Accuracy vs. Generating Density")
end

function plot_enrichment_fold(ticks, measure_res)
    bars = average_iteration_bars(ticks, measure_res,
                                  preprocess = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Network Enrichment Fold",
               xlabel = "Generating Network Density",
               title = "Network Enrichment vs. Generating Density")
end

function plot_test_ll(ticks, measure_res)
    bars = average_iteration_bars(ticks, measure_res)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Test Log-Likelihood",
               xlabel = "Generating Network Density",
               title = "Test Log-Likelihood vs. Generating Density")
end

function enrichment_curve(truths)
    curve = Array(Float64, length(truths))
    total_true = 0
    total_claimed = 0
    for i = 1:length(truths)
        val = truths[i]

        total_claimed = i
        if val
            total_true += 1;
        end

        curve[i] = total_true / total_claimed
    end
    curve
end

function plot_enrichment_curve(ticks, measure_res)
    (n_models, n_gens, n_iters) = size(measure_res)
    n_states = length(measure_res[1,1,1])
    model_ixs = [2, 3]

    gen_ix = 3
    gen_name = ticks[2][gen_ix]

    lines = []
    for model_ix = model_ixs
        model_name = ticks[1][model_ix]
        model_curves = []
        for iteration_ix = 1:n_iters, state_ix = 1:n_states
            # Calculate an enrichment curve from edge matching data
            curve = enrichment_curve(measure_res[model_ix, gen_ix, iteration_ix][state_ix])
            push!(model_curves, curve)

            # Create plot element line and add it to the list
            curve_line = line(curve,
                              color=colors[model_ix],
                              alpha=0.4)
            push!(lines, curve_line)
        end

        # Calculate the average curve for this model and push it
        avg_curve = mean(reduce(hcat, model_curves), 2)
        avg_line = line(avg_curve,
                        model_name,
                        color=colors[model_ix], alpha = 1.0, linewidth=4)
        push!(lines, avg_line)
    end

    # Generate a figure with all of the lines
    fig = axis(lines...,
               legend = "upper right",
               ylabel = "Proportion correct",
               xlabel = "Number of edges predicted",
               title = "Enrichment Curve ($gen_name)",
               ylims = (0.0, 1.05))
end

function convert_mean(m, args...; kwargs...)
    mean(convert(Array{Float64}, m), args...; kwargs...)
end

plotting_functions = Dict("edge_accuracy" => plot_edge_accuracy,
                          "label_accuracy" => plot_label_accuracy,
                          "test_ll" => plot_test_ll,
                          "enrichment_fold" => plot_enrichment_fold,
                          "edge_matches" => plot_enrichment_curve)

colors = [
          "#3366CC", "#DC3912", "#FF9902", "#0C9618", "#0099C6",
          "#990099", "#DD4477", "#66AA00", "#B82E2E", "#316395",
          "#994499", "#22AA99", "#AAAA11", "#6633CC", "#E67300"
          ]
end

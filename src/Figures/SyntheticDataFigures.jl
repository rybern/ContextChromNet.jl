module SyntheticDataFigures

using PyPlot.(savefig)
using SimplePlot
using HypothesisTests
using ArgParse
using DataStructures

function build_figures_v1(results_file,
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

function build_figures(output_directory,
                       results_file = "saved_outputs/3-23-16-synth-test.dump")
    (vars, ticks, res, specs) = open(deserialize, results_file)
    vars = collect(vars)
    ticks = collect(ticks)

    plot_test_ll_vs_model_k(vars, ticks, res, specs)
end


function test_result_bars(results_file = "saved_outputs/3-23-16-synth-test.dump")
    (vars, ticks, res, specs) = open(deserialize, results_file)

    vars = collect(vars)
    ticks = collect(ticks)

    value_map = Dict("generator_density" => 1,
                     "generator_shape" => 1,
                     "validation_measure" => 2)

    result_bars(vars, ticks, res,
                value_map,
                "num_model_states",
                "distribution",
                mean)
end

function result_bars(vars,
                     ticks,
                     res,
                     var_value_map,
                     independant_var,
                     line_var;
                     process_final_shape = convert_mean,
                     process_elements = identity)
    res_inds = map(var -> var in keys(var_value_map) ? var_value_map[var] : (:), vars)
    tensor = slice(res, res_inds...)

    old_indep_ind = findfirst(vars, independant_var)
    old_line_ind = findfirst(vars, line_var)

    vars_not_in_map = [!(var in keys(var_value_map)) for var in vars]
    inds_left = collect(1:length(size(res)))[vars_not_in_map]
    indep_ind = findfirst(inds_left, old_indep_ind)
    line_ind = findfirst(inds_left, old_line_ind)

    tensor_bars(tensor,
                ticks[old_indep_ind],
                ticks[old_line_ind],
                indep_ind,
                line_ind,
                process_final_shape = process_final_shape,
                process_elements = process_elements)
end

function tensor_bars(tensor,
                     xticks,
                     line_ticks,
                     independant_var,
                     line_var;
                     process_elements = identity,
                     process_final_shape = identity)
    bars = [bar(xticks,
                [process_final_shape(map(process_elements, (slicedim(slicedim(tensor, line_var, line_ix),
                                                                     independant_var,
                                                                     ind_ix))))
                 for ind_ix = 1:size(tensor)[independant_var]],
                line_name,
                color = colors[line_ix])
            for (line_ix, line_name) = enumerate(line_ticks)]
end

function runPlots()
    plots = []

    addPlot = (dump, plotFn) -> push!(plots, () -> plotFn(collect(dump[1]), collect(dump[2]), dump[3], dump[4]))

    vary_model = open(deserialize, "saved_outputs/synth-vary-model-3-29-16.dump")
    vary_shape = open(deserialize, "saved_outputs/synth-vary-shape-3-26-16.dump")
    vary_shape1 = open(deserialize, "saved_outputs/synth-vary-shape-mk1-3-29-16.dump")
#    vary_density = open(deserialize, "saved_outputs/3-23-16-synth-test.dump")
#    vary_density = open(deserialize, "saved_outputs/synth-full-2-26-16-final.dump")

    # number of model states
    addPlot(vary_model, plot_test_ll_vs_model_k)
    addPlot(vary_model, plot_active_states_vs_model_k)

    # number of shape states, default model state
    addPlot(vary_shape, plot_label_accuracy_vs_shape)
    addPlot(vary_shape, plot_enrichment_vs_shape)

    # number of shape states, 1 model state
    addPlot(vary_shape1, plot_label_accuracy_vs_shape)
    addPlot(vary_shape1, plot_enrichment_vs_shape)

    # density of generating states
#    addPlot(vary_density, plot_edge_accuracy_vs_density)
#    addPlot(vary_density, plot_enrichment_vs_density)
#    addPlot(vary_density, plot_label_accuracy_vs_density)

#    addPlot(vary_density, plot_test_ll_vs_density)

    plots
end

function plot_edge_accuracy_vs_density(vars, ticks, res, specs;
                                       model_k_ind = 1,
                                       shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          EDGE_ACCURACY_MEASURE)),
                       DENSITY_VARIABLE,
                       DISTRIBUTION_VARIABLE,
                       process_elements = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Edge Accuracy",
               xlabel = "Generating Network Density",
               title = "Edge Accuracy vs. Generating Density")
end

function plot_label_accuracy_vs_density(vars, ticks, res, specs;
                                       model_k_ind = 1,
                                       shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          LABEL_ACCURACY_MEASURE)),
                       DENSITY_VARIABLE,
                       DISTRIBUTION_VARIABLE,
                       process_elements = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Label Accuracy",
               xlabel = "Generating Network Density",
               title = "Label Accuracy vs. Generating Density")
end

function plot_test_ll_vs_density(vars, ticks, res, specs;
                                       model_k_ind = 1,
                                       shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          TEST_LL_MEASURE)),
                       DENSITY_VARIABLE,
                       DISTRIBUTION_VARIABLE,
                       process_elements = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Test Log-Likelihood",
               xlabel = "Generating Network Density",
               title = "Test Log-Likelihood vs. Generating Density")
end

function plot_enrichment_vs_density(vars, ticks, res, specs;
                                       model_k_ind = 1,
                                       shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          ENRICHMENT_FOLD_MEASURE)),
                       DENSITY_VARIABLE,
                       DISTRIBUTION_VARIABLE,
                       process_elements = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Enrichment Fold",
               xlabel = "Generating Network Density",
               title = "Enrichment Fold vs. Generating Density")
end

function plot_label_accuracy_vs_shape(vars, ticks, res, specs;
                                     model_k_ind = 1,
                                     density_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            DENSITY_VARIABLE => density_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          LABEL_ACCURACY_MEASURE)),
                       SHAPE_VARIABLE,
                       DISTRIBUTION_VARIABLE)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Label Accuracy",
               xlabel = "Generating Network Shape",
               title = "Label Accuracy vs. Generating Shape")
end

function plot_enrichment_vs_shape(vars, ticks, res, specs;
                                  model_k_ind = 1,
                                  density_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(MODEL_K_VARIABLE => model_k_ind,
                            DENSITY_VARIABLE => density_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          ENRICHMENT_FOLD_MEASURE)),
                       SHAPE_VARIABLE,
    DISTRIBUTION_VARIABLE,
    process_elements = convert_mean)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Enrichment",
               xlabel = "Generating Network Shape",
               title = "Enrichment vs. Generating Shape")
end

function plot_active_states_vs_model_k(vars, ticks, res, specs,
                                       density_ind = 1,
                                       shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(DENSITY_VARIABLE => density_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          ACTIVE_STATES_MEASURE)),
                       MODEL_K_VARIABLE,
                       DISTRIBUTION_VARIABLE)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Avg. # Active States",
               xlabel = "Number of Model States",
               title = "Active States vs. Model K (True K: $(specs[3]))")
end

function plot_test_ll_vs_model_k(vars, ticks, res, specs,
                                 density_ind = 1,
                                 shape_ind = 1)
    bars = result_bars(vars,
                       ticks,
                       res,
                       Dict(DENSITY_VARIABLE => density_ind,
                            SHAPE_VARIABLE => shape_ind,
                            MEASURE_VARIABLE => findfirst(ticks[findfirst(vars, MEASURE_VARIABLE)],
                                                          TEST_LL_MEASURE)),
                       MODEL_K_VARIABLE,
                       DISTRIBUTION_VARIABLE)

    fig = axis(bars...,
               legend = "upper right",
               ylabel = "Test-Log Likelihood",
               xlabel = "Number of Model States",
               title = "Test LL vs. Model K (True K: $(specs[3]))")
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

function truncate_generator_names(generator_names)
    [name[length(name)-2:length(name)] for name = generator_names]
end

function convert_mean(m, args...; kwargs...)
    mean(convert(Array{Float64}, m), args...; kwargs...)
end

# measure_names
EDGE_ACCURACY_MEASURE = "edge_accuracy"
LABEL_ACCURACY_MEASURE = "label_accuracy"
ACTIVE_STATES_MEASURE = "num_active_states"
TEST_LL_MEASURE = "test_ll"
ENRICHMENT_FOLD_MEASURE = "enrichment_fold"
EDGE_MATCHES_MEASURE = "edge_matches"
all_measures = [EDGE_ACCURACY_MEASURE,
                LABEL_ACCURACY_MEASURE,
                ACTIVE_STATES_MEASURE,
                TEST_LL_MEASURE,
                ENRICHMENT_FOLD_MEASURE,
                EDGE_MATCHES_MEASURE]

# variable names
DISTRIBUTION_VARIABLE = "distribution"
MODEL_K_VARIABLE = "num_model_states"
DENSITY_VARIABLE = "generator_density"
SHAPE_VARIABLE = "generator_shape"
MEASURE_VARIABLE = "validation_measure"
ITERATION_VARIABLE = "iteration"
all_variables = [DISTRIBUTION_VARIABLE,
                 MODEL_K_VARIABLE,
                 DENSITY_VARIABLE,
                 SHAPE_VARIABLE,
                 MEASURE_VARIABLE,
                 ITERATION_VARIABLE]

colors = [
          "#3366CC", "#DC3912", "#FF9902", "#0C9618", "#0099C6",
          "#990099", "#DD4477", "#66AA00", "#B82E2E", "#316395",
          "#994499", "#22AA99", "#AAAA11", "#6633CC", "#E67300"
          ]
end

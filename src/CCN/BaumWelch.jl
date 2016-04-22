module BaumWelch
export baum_welch, log_likelihood

using BaumWelchConvergence
using BaumWelchUtils
using EmissionDistributions
using HMMTypes
using ExtendedLogs
using SharedEmissions
using Compat

using SimpleLogging
using ArrayViews

function toy()
    data = rand(3, 100)
    k = 2
    (estimate, model, ll) = baum_welch(10, data, k, is_converged = ll_convergence(.01), verbose = 1)
    labels = gamma_to_labels(estimate.gamma)
    (labels, ll)
end

function baum_welch(num_runs :: Integer,
                            args...;
                            verbose = Void, # :: Union{Type{Void}, Integer} = 0,
                            result_writer = Void, # :: Union{Type{Void}, Function} = Void,
                            kwargs...)
    logstrln("Running Baum-Welch", verbose)
    flush(STDOUT)
    function run(i)
        if verbose != Void
            logstrln("BW Random Restart $i/$num_runs", verbose)
            (estimate, model, ll) = baum_welch(args...; verbose = verbose + 1, kwargs...)
        else
            (estimate, model, ll) = baum_welch(args...; verbose = Void, kwargs...)
        end

        if result_writer != Void
            result_writer("run_$i", estimate, model, ll)
        end

        (estimate, model, ll)
    end

    runs = map(run, 1:num_runs)



  #  best = runs[1]

 #   if result_writer != Void
 #       result_writer("best", best...)
  #  end

    if verbose != Void
        logstrln("$num_runs restarts complete",
                 verbose)
    end
    sort!(runs, by = run -> run[3], rev = true)
    runs[1]
end


function baum_welch{N <: Number}(data :: DenseArray{N, 2},
                    k :: Integer,
                    fit_emissions = fit_full_cov :: Function, # data -> gamma -> State
                    emission_log_pdf = dist_log_pdf :: Function;# weights -> state -> data -> log probability
                    initial_model = HMMStateModel(fit_states_to_labels(rand(1:k, size(data, 2)),
                                                                       k,
                                                                       data,
                                                                       fit_emissions),
                                                  ones(k, k) / k),
                    # is_converged :: gamma, states, ll,
                    #                 gamma, states, ll,
                    #                 iteration ->
                    #                 Bool
                    is_converged = ll_convergence(.001) :: Function,
#                    is_converged = iteration_convergence(5) :: Function,
                    verbose :: Union{Type{Void}, Integer} = Void)
    if k == 1
        gamma = ones(1, size(data, 2))
        transition = ones(1, 1)
        states = collect(fit_emissions(data, gamma))
        model = HMMStateModel(states, transition)
        ll = log_likelihood(data, k, model, emission_log_pdf)
        return (HMMEstimate(gamma), HMMStateModel(states, transition), ll)
    end


    logstr("Initial data sharing ... ", verbose)
    data = convert(SharedArray, data)
    logstrln("done", verbose == Void ? Void : 0, false)

    spec = ProblemSpec(size(data, 2),
                       size(data, 1),
                       k)

    emissions_flipped = SharedArray(Float64, spec.n, spec.k)
    emissions = SharedArray(Float64, spec.k, spec.n)
    function emission_log_density(t)
        view(emissions, :, t)
    end

    logstr("Initial emission calculation ... ", verbose)
    update_emissions!(emission_log_pdf, initial_model.states, data, emissions, emissions_flipped)
    logstrln("done", verbose == Void ? Void : 0, false)

    initial = ones(k) / k
    log_initial = eln_arr(initial)
    transition = initial_model.trans
    log_transition = eln_arr(transition)

    oldStates = initial_model.states
    oldLL = -Inf
    ll = -Inf
    oldGamma = false;


    logstrln("Starting EM", verbose)

    iteration = 1;
    while true
        logstrln("Starting iteration $iteration", verbose)

        logstrln("Starting E Step", verbose == Void ? Void : verbose + 1)
        (log_alpha, log_beta, newGammaPromise) = bw_e_step(spec,
                                                           log_transition,
                                                           emission_log_density,
                                                           log_initial)
        logstrln("Finished E Step", verbose == Void ? Void : verbose + 1)

        logstrln("Starting M Step", verbose == Void ? Void : verbose + 1)
        (newTransition, newInitial, newStates) = bw_m_step(spec,
                                                           log_transition,
                                                           emission_log_density,
                                                           fit_emissions,
                                                           oldStates,
                                                           data,
                                                           log_alpha,
                                                           log_beta,
                                                           newGammaPromise,
                                                           verbose)
        logstrln("Ending M Step", verbose == Void ? Void : verbose + 1)

        newGamma = fetch(newGammaPromise)


        logstr("Log-Like ... ", verbose == Void ? Void : verbose + 1)
        ll = log_likelihood(log_alpha)
        logstrln("done", verbose == Void ? Void : 0, false)

        converged = iteration != 1 && is_converged(oldGamma, oldStates, oldLL,
                                                   newGamma, newStates, ll,
                                                   iteration)


        logstr("Emissions update ... ", verbose == Void ? Void : verbose + 1)
        update_emissions!(emission_log_pdf, newStates, data, emissions, emissions_flipped)
        logstrln("done", verbose == Void ? Void : 0, false)

        initial = newInitial
        log_initial = eln_arr(initial)

        transition = newTransition
        log_transition = eln_arr(transition)

        oldStates = newStates
        oldGamma = newGamma

        oldLL = ll

        if verbose != Void
            logstrln("Iteration complete; log-likelihood: $ll", verbose + 1)
        end

        converged && break

        iteration = iteration + 1
    end

    if verbose != Void
        logstrln("Converged", verbose)
    end

    (HMMEstimate(oldGamma), HMMStateModel(oldStates, transition), ll)

end


function log_likelihood(data,
                         k,
                         model,
                         emission_log_pdf)
    (p, n) = size(data)
    spec = ProblemSpec(n, p, k)

    initial = ones(k) / k
    log_initial = log(initial);

    log_transition = eln_arr(model.trans);

    data = convert(SharedArray, data)

    emissions_flipped = SharedArray(Float64, spec.n, spec.k)
    emissions = SharedArray(Float64, spec.k, spec.n)
    update_emissions!(emission_log_pdf, model.states, data, emissions, emissions_flipped)

    function emission_log_density(t)
        view(emissions, :, t)
    end

    log_alpha = forward(spec, log_transition, emission_log_density, log_initial);
    log_likelihood(log_alpha)
end

function log_likelihood(log_alpha :: Array{Float64, 2}) # k x n
    # sum of the probabilities of the paths
    elnsum_arr(log_alpha[:, end]);
end


function bw_e_step(spec :: ProblemSpec,
                   log_transition :: Array{Float64, 2},
                   emission_log_density :: Function, # Int -> log pdf
                   log_initial :: Array{Float64, 1})
    log_alpha_thread = @spawn forward(spec, log_transition, emission_log_density, log_initial)
    log_beta = backward(spec, log_transition, emission_log_density)
    log_alpha = fetch(log_alpha_thread)
    gamma_promise = @spawn gamma(spec, log_alpha, log_beta)

    (log_alpha, log_beta, gamma_promise)
end

function backward(spec,
                  log_transition,
                  emission_log_density)

    log_beta = Array(Float64, spec.k, spec.n);

    log_beta[:, spec.n] = 0;

    for t = spec.n-1:-1:1
        nextDist = emission_log_density(t+1) + log_beta[:, t+1]
        for i = 1:spec.k
            elems = log_transition[i, :]' + nextDist
            log_beta[i, t] = elnsum_arr(elems)
        end
    end

    log_beta
end

function forward(spec,
                 log_transition,
                 emission_log_density,
                 log_initial)


    log_alpha = Array(Float64, spec.k, spec.n);

    em = emission_log_density(1);

    log_alpha[:, 1] = log_initial + em

    for t = 2:spec.n
        log_alpha[:, t] = emission_log_density(t)
        for i = 1:spec.k
            elems = log_transition[:, i] + log_alpha[:, t-1]
            log_alpha[i, t] += elnsum_arr(elems)
        end
    end

    log_alpha
end

function gamma(spec, log_alpha, log_beta)
    gma = Array(Float64, spec.k, spec.n)

    innerProducts = log_alpha + log_beta

    for t = 1:spec.n
        gma[:, t] = eexp_arr(innerProducts[:, t] - elnsum_arr(innerProducts[:, t]))
    end

    gma
end

function bw_m_step{N <: Number}(spec :: ProblemSpec,
                   log_transition :: Array{Float64, 2},
                   emission_log_density :: Function, # Int -> log pdf
                   fit_emissions :: Function, # data -> weights -> old_states -> new_states
                   states :: Array{HMMState, 1},
                   data :: AbstractArray{N, 2},
                   log_alpha :: Array{Float64, 2},
                   log_beta :: Array{Float64, 2},
                   gamma_promise,  # promise of Array{Float, 2}
                   verbose :: Union{Type{Void}, Integer})

    logstr("Transition matrix ... ", verbose == Void ? Void : verbose + 2)
    newTransition = updateTransitionMatrix(spec,
                                           log_transition,
                                           emission_log_density,
                                           log_alpha,
                                           log_beta,
                                           gamma_promise)
    logstrln("done", verbose == Void ? Void : 0, false)

    gma = fetch(gamma_promise)
    newInitial = gma[:, 1]

    logstr("Fitting states ... ", verbose == Void ? Void : verbose + 2)
    newStates = fit_emissions(data, gma, states);
    logstrln("done", verbose == Void ? Void : 0, false)

    (newTransition, newInitial, newStates)
end

function updateTransitionMatrix(spec,
                                log_transition,
                                emission_log_density,
                                log_alpha,
                                log_beta,
                                gammaPromise)

    logEpsSum = @parallel (x, y)->map(elnsum, x, y) for t = 1:spec.n-1
        # accumulate the log of sums using elnsum
        logEpsT(spec, log_transition, emission_log_density, log_alpha, log_beta, t)
    end

    epsSum = map(eexp, logEpsSum)
    gamma = fetch(gammaPromise)
    gammaSum = sum(gamma, 2) - gamma[:, spec.n] + 1;
    newTrans = zeros(spec.k, spec.k);
    for i = 1:spec.k
        newTrans[i, :] = epsSum[i, :] / gammaSum[i];
    end

    newTrans
end

function logEpsT(spec,
                 log_transition,
                 emission_log_density,
                 log_alpha,
                 log_beta,
                 t)

    log_betaWeighted = reshape(log_beta[:, t+1] + emission_log_density(t+1), (1, spec.k))

    # (1 x k) .+ (k x k) .+ (k x 1)
    # the former is added to the matrix by repeating rows
    # the latter by repeating columns
    numerator = log_betaWeighted .+ log_transition .+ log_alpha[:, t];
    normalizer = elnsum_arr(log_alpha[:, t] + log_beta[:, t])

    numerator - normalizer
end



function update_emissions!(emission_log_pdf, states, data, emissions, emissions_flipped)
    parallel_emissions!(data, emissions, emissions_flipped, states, emission_log_pdf)
end

function fit_states_to_labels{N <: Number}(labels :: Array{Int, 1},
                                            k :: Integer,
                                            data :: Array{N, 2},
                                            # data -> weights -> new_states
                                            fit_emissions :: Function)
    fit_emissions(data, labels_to_gamma(labels, k))
end

end

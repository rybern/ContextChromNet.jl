module BaumWelch
export baum_welch

using BaumWelchConvergence
using BaumWelchUtils
using EmissionDistributions
using HMMTypes
using ExtendedLogs
using SharedEmissions

using ArrayViews

function toy()
    data = rand(3, 10)
    k = 2
    (estimate, model, ll) = baum_welch(data, k, is_converged = ll_convergence(.01))
    labels = gamma_to_labels(estimate.gamma)
end

function baum_welch{N <: Number} (data :: Array{N, 2},
                    k :: Integer,
                    # fit_emissions :: data -> gamma -> State
                    fit_emissions = fit_full_cov :: Function,
                    # emission_pdf :: data -> gamma -> log probability
                    emission_pdf = dist_log_pdf :: Function;
                    # intial_model :: (states, transition)
                    initial_model = HMMStateModel(fit_states_to_labels(rand(1:k, size(data, 2)),
                                                                       k,
                                                                       data,
                                                                       fit_emissions),
                                                  ones(k, k) / k),
                    # is_converged :: gamma, states, ll,
                    #                 gamma, states, ll,
                    #                 iteration ->
                    #                 Bool
                    is_converged = iteration_convergence(5) :: Function)

    data = convert(SharedArray, data)

    spec = ProblemSpec(size(data, 2),
                       size(data, 1), 
                       k)

    emissionsFlipped = SharedArray(Float64, spec.n, spec.k)
    emissions = SharedArray(Float64, spec.k, spec.n)
    function emissionFn(t)
        view(emissions, :, t)
    end

    updateEmissions!(emission_pdf, initial_model.states, data, emissions, emissionsFlipped)

    initial = ones(k) / k
    logInitial = eln_arr(initial)

    transition = initial_model.trans
    logTransition = eln_arr(transition)

    oldStates = initial_model.states
    oldLL = -Inf
    ll = -Inf
    oldGamma = false;

    iteration = 1;
    while true
        (fwd, bwd, newGammaPromise) = bw_e_step(spec, logTransition, emissionFn, logInitial)

        (newTransition, newInitial, newStates) = bw_m_step(spec,
                                                                logTransition, 
                                                                emissionFn, 
                                                                fit_emissions,
                                                                oldStates,
                                                                data,
                                                                fwd,
                                                                bwd,
                                                                newGammaPromise)

        newGamma = fetch(newGammaPromise)

        ll = logLikelihood(spec, fwd)

        converged = iteration != 1 && is_converged(oldGamma, oldStates, oldLL,
                                                   newGamma, newStates, ll,
                                                   iteration)

        updateEmissions!(emission_pdf, newStates, data, emissions, emissionsFlipped)

        initial = newInitial
        logInitial = eln_arr(initial)

        transition = newTransition
        logTransition = eln_arr(transition)
        
        oldStates = newStates
        oldGamma = newGamma

        oldLL = ll

        println("iteration $iteration: log-likelihood $ll")

        converged && break
        
        iteration = iteration + 1
    end

    (HMMEstimate(initial, oldGamma), HMMStateModel(oldStates, transition), ll)

end


function logLikelihood(spec,
                       data,
                       model,
                       logEmissionDist,
                       initial = vec(ones(spec.k) / spec.k))

    logTransition = eln_arr(model.trans);

    data = convert(SharedArray, data)
    emissionsFlipped = SharedArray(Float64, spec.n, spec.k)
    emissions = SharedArray(Float64, spec.k, spec.n)
    function emissionFn(t)
        view(emissions, :, t)
    end

    updateEmissions!(logEmissionDist, model.states, data, emissions, emissionsFlipped)
    logInitial = log(initial);

    fwd = forward(spec, logTransition, emissionFn, logInitial);
   
    # sum of the probabilities of the paths converging to the final
    # state: the transition from any state to the final states is
    # uniform
    logLikelihood(spec, fwd)
end

function logLikelihood(spec, logAlpha)
    elnsum_arr(logAlpha[end, :]) - log(spec.p);
end


function bw_e_step(spec, logTransition, logEmission, logInitial)
    fwd_thread = @spawn forward(spec, logTransition, logEmission, logInitial)
    bwd = backward(spec, logTransition, logEmission)
    fwd = fetch(fwd_thread)
    gmaPromise = @spawn gamma(spec, fwd, bwd)

    (fwd, bwd, gmaPromise)
end

function backward(spec,
                  logTransition,
                  logEmission)

    logBeta = Array(Float64, spec.k, spec.n);

    logBeta[:, spec.n] = 0;

    for t = spec.n-1:-1:1
        nextDist = logEmission(t+1) + logBeta[:, t+1]
        for i = 1:spec.k
            elems = logTransition[i, :]' + nextDist
            logBeta[i, t] = elnsum_arr(elems)
        end
    end

    logBeta
end

function forward(spec, 
                 logTransition,
                 logEmission,
                 logInitial)

    logAlpha = Array(Float64, spec.k, spec.n);

    em = logEmission(1);

    logAlpha[:, 1] = logInitial + em

    for t = 2:spec.n
        logAlpha[:, t] = logEmission(t)
        for i = 1:spec.k
            elems = logTransition[:, i] + logAlpha[:, t-1]
            logAlpha[i, t] += elnsum_arr(elems)
        end
    end

    logAlpha
end

function gamma(spec, logAlpha, logBeta)
    gma = Array(Float64, spec.k, spec.n)

    innerProducts = logAlpha + logBeta

    for t = 1:spec.n
        gma[:, t] = eexp_arr(innerProducts[:, t] - elnsum_arr(innerProducts[:, t]))
    end

    gma
end

function bw_m_step(spec, logTransition, logEmission, updateStates, states, data, fwd, bwd, gmaProm)
    newTransition = updateTransitionMatrix(spec, logTransition, logEmission, fwd, bwd, gmaProm)
    gma = fetch(gmaProm)
    newInitial = gma[:, 1]
    newStates = updateStates(data, gma, states);
    
    (newTransition, newInitial, newStates)
end

function updateTransitionMatrix(spec,
                                logTransition,
                                logEmission,
                                logAlpha,
                                logBeta,
                                gammaPromise)

    logEpsSum = @parallel (x, y)->map(elnsum, x, y) for t = 1:spec.n-1
        # accumulate the log of sums using elnsum
        logEpsT(spec, logTransition, logEmission, logAlpha, logBeta, t)
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
                 logTransition,
                 logEmission,
                 logAlpha,
                 logBeta,
                 t)

    logBetaWeighted = reshape(logBeta[:, t+1] + logEmission(t+1), (1, spec.k))
    
    # (1 x k) .+ (k x k) .+ (k x 1)
    # the former is added to the matrix by repeating rows
    # the latter by repeating columns
    numerator = logBetaWeighted .+ logTransition .+ logAlpha[:, t];
    normalizer = elnsum_arr(logAlpha[:, t] + logBeta[:, t])

    numerator - normalizer
end



function updateEmissions!(logEmissionDist, states, data, emissions, emissionsFlipped)
    parallel_emissions!(data, emissions, emissionsFlipped, states, logEmissionDist)
end

function fit_states_to_labels{N <: Number} (labels :: Array{Int, 1},
                                            k :: Integer,
                                            data :: Array{N, 2},
                                            fit_emissions :: Function)
    fit_emissions(data, labels_to_gamma(labels, k))
end



end



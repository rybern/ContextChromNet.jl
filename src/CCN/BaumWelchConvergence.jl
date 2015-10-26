module BaumWelchConvergence
export delta_convergence, iteration_convergence, ll_convergence

# convergence :: gamma, states, ll ->
#                gamma, states, ll ->
#                iteration ->
#                Bool

function delta_convergence (max_gamma_delta,
                            max_cov_delta,
                            max_mean_delta)

    function converged (oldGamma, oldStates, oldLL,
                        newGamma, newStates, newLL,
                        iteration)
        oldK = size(newStates)[1]
        activeIndices = filter(i -> newStates[i].active, 1:oldK)
        activeNewStates = newStates[activeIndices]
        activeOldStates = oldStates[activeIndices]
        k = size(activeIndices)[1]

        muConv = all(map(i -> maxDeltaError(mean(activeNewStates[i].dist),
                                            mean(activeOldStates[i].dist),
                                            max_mean_delta),
                         [1:k]))
        covConv = all(map(i -> maxDeltaError(cov(activeNewStates[i].dist),
                                             cov(activeOldStates[i].dist),
                                             max_cov_delta),
                             [1:k]))
        muConv && covConv
    end
end

function iteration_convergence (halting_iteration)
    function converged (oldGamma, oldStates, oldLL,
                        newGamma, newStates, newLL,
                        iteration)
        iteration > halting_iteration
    end
end

function ll_convergence (max_percent_change)
    function converged (oldGamma, oldStates, oldLL,
                        newGamma, newStates, newLL,
                        iteration)
        percent_changed = abs((newLL - oldLL) / oldLL)
        println(percent_changed)
        percent_changed < max_percent_change || oldLL > newLL
    end
end

function maxDeltaError(new, old, eps)
    maximum(abs((new - old) ./ old)) <= eps
end

end

module StateMatching
export match_states

using BaumWelchUtils
using HMMTypes
using Munkres

function test_match_gamma_even()
    # generate data, random arrangement of states
    true_labels = rand(1:10, 100)
    true_to_found = shuffle(collect(1:10))

    found_labels = [true_to_found[t] for t = true_labels]
    found_gamma = labels_to_gamma(found_labels, 10)

    # compute permutation of rearranged states
    new_to_old = label_permutation(found_labels, 10,
                                   true_labels, 10)
    # permute found states back to original labels
    permuted_gamma = permute_gamma(new_to_old, found_gamma)
    permuted_labels = gamma_to_labels(permuted_gamma)

    println(new_to_old)
    println(true_to_found)

    permuted_labels == true_labels
end

# this turns out to be undefined in general, but it's only being used for label matching, so just return the permuted labels
function match_states(estimate, found_model, true_labels, true_model)
    found_k = size(found_model.trans, 1)
    true_k = size(true_model.trans, 1)

    found_labels = gamma_to_labels(estimate.gamma)
    new_to_old = label_permutation(found_labels, found_k,
                                   true_labels, true_k)
    apply_permutation(new_to_old, estimate, found_model)
end

# Determine the permutation of labels such that assigning old_label = perm[new_label] maximizes label accuracy
function label_permutation(found_labels, found_k,
                           true_labels, true_k)
    confusion_matrix = label_confusion_matrix(found_labels, found_k,
                                              true_labels, true_k)

    label_permutation(confusion_matrix)
end

function label_permutation(confusion_matrix)
    munkres(-confusion_matrix')
end

function permute_gamma(new_to_old, old_gamma,
                       new_p = length(new_to_old),
                       n = size(old_gamma, 2))
    nonzero = new_to_old .> 0
    zeros(Float64, new_p, n)[nonzero, :] = old_gamma[new_to_old[nonzero], :]
end

function permute_trans(new_to_old, old_trans,
                       new_p = length(new_to_old))
    nonzero = new_to_old .> 0
    zeros(Float64, new_p, new_p)[nonzero, nonzero] = old_trans[new_to_old[nonzero], new_to_old[nonzero]]
end

function permute_states(new_to_old, old_states,
                        new_p = length(new_to_old))
    nonzero = new_to_old .> 0
    [HMMState(nothing, false) for i = 1:new_p][nonzero] = old_states[new_to_old[nonzero]]
end

function permute_results(new_to_old, estimate, model)
    (HMMEstimate(permute_gamma(new_to_old, estimate.gamma)),
     HMMStateModel(permute_states(new_to_old, model.states),
                   permute_trans(new_to_old, model.trans)))
end

function apply_permutation_(new_to_old, estimate, model)
    (p, n) = size(estimate.gamma)
    k = size(model.trans, 1)

    new_gamma = Array(Float64, p, n)

    for i = 1:n
        for j = 1:k
            new_gamma[j, i] = estimate.gamma[new_to_old[j], i]
        end
    end

    new_trans = Array(Float64, k, k)
    new_states = Array(HMMState, k)

    for i = 1:k
        for j = 1:k
            new_trans[i, j] = model.trans[new_to_old[i], new_to_old[j]]
        end

        new_states[i] = model.states[new_to_old[i]]
    end

    (HMMEstimate(new_gamma), HMMStateModel(new_states, new_trans))
end


function match_states_(estimate, found_model, true_labels, true_model)
    k = size(found_model.trans, 1)
    permutation = optimal_label_permutation(gamma_to_labels(estimate.gamma), true_labels, k)
    apply_permutation(permutation, estimate, found_model)
end

function greedy_label_permutation(labels1, labels2, k)
    # greedy matching. maybe optimal? probably close.
    permutation = Array(Int, k)

    sorted_by_abundance = sort(collect(1:k), by = l -> sum(labels1 .== l), rev = true)

    possible_matches = collect(1:k)
    for l = sorted_by_abundance
        to_match = labels1 .== l

        match_scores = [sum(to_match & (labels2 .== match))
                        for match = possible_matches]
        max_index = indmax(match_scores)
        permutation[l] = possible_matches[max_index]

        labels1 = labels1[!to_match]
        labels2 = labels2[!to_match]

        splice!(possible_matches, max_index)
    end

    permutation
end

function optimal_label_permutation(labels1, labels2, k)
    perms = permutations(1:k)
    max_score = 0
    max_perm = Void

    temp_labels = Array(Int64, size(labels1))
    for perm = perms
        for i = 1:length(labels1)
            temp_labels[i] = perm[labels1[i]]
        end

        score = sum(temp_labels .== labels2)

        if max_perm == Void || score > max_score
            max_perm = perm
            max_score = score
        end
    end

    max_perm
end

function apply_permutation(old_to_new, estimate, model)
    (k, n) = size(estimate.gamma)
    new_to_old = [findfirst(old_to_new, i) for i = 1:k]

    new_gamma = Array(Float64, k, n)

    for i = 1:n
        for j = 1:k
            new_gamma[j, i] = estimate.gamma[new_to_old[j], i]
        end
    end

    new_trans = Array(Float64, k, k)
    new_states = Array(HMMState, k)

    for i = 1:k
        for j = 1:k
            new_trans[i, j] = model.trans[new_to_old[i], new_to_old[j]]
        end

        new_states[i] = model.states[new_to_old[i]]
    end

    (HMMEstimate(new_gamma), HMMStateModel(new_states, new_trans))
end

end

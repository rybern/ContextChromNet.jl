module StateMatching
export match_states

using BaumWelchUtils
using HMMTypes

function match_states(estimate, found_model, true_labels, true_model)
    k = size(found_model.trans, 1)
    permutation = optimal_label_permutation_(gamma_to_labels(estimate.gamma), true_labels, k)
    apply_permutation(permutation, estimate, found_model)
end

function optimal_label_permutation_(labels1, labels2, k)
    # greedy matching. maybe optimal? probably close.
    permutation = Array(Int, k)

    sorted_by_abundance = sort([1:k], by = l -> sum(labels1 .== l), rev = true)

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
    max_perm = Nothing

    temp_labels = Array(Int64, size(labels1))
    for perm = perms
        for i = 1:length(labels1)
            temp_labels[i] = perm[labels1[i]]
        end

        score = sum(temp_labels .== labels2)

        if max_perm == Nothing || score > max_score
            max_perm = perm
            max_score = score            
        end
    end

    max_perm
end

function apply_permutation (old_to_new, estimate, model)
    (p, n) = size(estimate.gamma)
    k = size(model.trans, 1)
    new_to_old = [findfirst(old_to_new, i) for i = 1:k]

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

end

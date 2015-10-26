module HMMTypes

export ProblemSpec, HMMState, HMMStateModel, HMMLabelModel, HMMCentroidModel, HMMEstimate

immutable ProblemSpec
    n :: Int64
    p :: Int64
    k :: Int64
end

immutable HMMState
    dist
    active :: Bool
end

immutable HMMStateModel
    states :: Array{HMMState, 1}
    trans :: Array{Float64, 2}
end

immutable HMMLabelModel
    gamma :: Array{Float64, 2}
    trans :: Array{Float64, 2}
end

immutable HMMCentroidModel
    indices :: Array{Int64, 1}
end

immutable HMMEstimate
    init :: Array{Float64, 1}
    gamma :: Array{Float64, 2}
end

end

module utils

export maybeCat, modelToMixture, scoop, dump, transpose!

function maybeCat(strA, strB)
    if ((strA != false) && (strB != false))
        string(strA, strB)
    else
        false
    end
end
        
function modelToMixture(model)
    [inv(cholfact(cov(s.dist))) for s = model.states]
end

function scoop(filepath)
    open(deserialize, filepath)
end

function dump(x,
              filepath = false)

    if (filepath != false)
        open(s -> serialize(s, x), filepath, "w");
    end
    
    x
end

function minInd(data; by = identity)
    extInd(data, indmin, by)
end

function maxInd(data; by = identity)
    extInd(data, indmax, by)
end

function extInd(data, op, by)
    op(map(by, data))
end

function gammaAssignments(gamma)
    map(indmax, map(i->gamma[:, i], [1:size(gamma, 2)]))
end

function fitTransitions(realTrans, foundTrans)
    n = size(realTrans)[1]
    perms = collect(permutations([1:n]))

    function transform(m, perm)
        permM = zeros(n, n)
        for i = 1:n
            permM[i, perm[i]] = 1
        end

        permM * m
    end
    
    function permError(perm)
        norm(realTrans - transform(foundTrans, perm))
    end
    print(map(permError, perms))
    minPermInd = indmin(map(permError, perms))
    perms[minPermInd]
end


## Transpose ##
const transposebaselength=64
function transpose!(B::StridedMatrix,A::StridedMatrix)
    m, n = size(A)
    size(B,1) == n && size(B,2) == m || throw(DimensionMismatch("transpose"))

    if m*n<=4*transposebaselength
        @inbounds begin
            for j = 1:n
                for i = 1:m
                    B[j,i] = transpose(A[i,j])
                end
            end
        end
    else
        transposeblock!(B,A,m,n,0,0)
    end
    return B
end
function transpose!(B::StridedVector, A::StridedMatrix)
    length(B) == length(A) && size(A,1) == 1 || throw(DimensionMismatch("transpose"))
    copy!(B, A)
end
function transpose!(B::StridedMatrix, A::StridedVector)
    length(B) == length(A) && size(B,1) == 1 || throw(DimensionMismatch("transpose"))
    copy!(B, A)
end
function transposeblock!(B::StridedMatrix,A::StridedMatrix,m::Int,n::Int,offseti::Int,offsetj::Int)
    if m*n<=transposebaselength
        @inbounds begin
            for j = offsetj+(1:n)
                for i = offseti+(1:m)
                    B[j,i] = transpose(A[i,j])
                end
            end
        end
    elseif m>n
        newm=m>>1
        transposeblock!(B,A,newm,n,offseti,offsetj)
        transposeblock!(B,A,m-newm,n,offseti+newm,offsetj)
    else
        newn=n>>1
        transposeblock!(B,A,m,newn,offseti,offsetj)
        transposeblock!(B,A,m,n-newn,offseti,offsetj+newn)
    end
    return B
end

end


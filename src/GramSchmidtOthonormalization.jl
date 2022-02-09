module GramSchmidtOthonormalization-

using LinearAlgebra

export classical_gram_schmidt, modified_gram_schmidt, modified_gram_schmidt_rutishauser

"""
    classical_gram_schmidt(matrix)

Apply the classical Gram-Schmidt orthogonalization process to the column vectors
of the input matrix `matrix`.

See also [modified_gram_schmidt](@ref), [modified_gram_schmidt_rutishauser](@ref)
"""
function classical_gram_schmidt(matrix)
    # orthogonalises the columns of the input matrix
    num_vectors = size(matrix)[2]
    orth_matrix = zeros(size(matrix))
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = matrix[:, vec_idx]
        sum = zeros(size(orth_matrix[:, 1]))
        for span_base_idx = 1:(vec_idx - 1)
            # compute sum
            sum += dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx]) * orth_matrix[:, span_base_idx]
        end
        orth_matrix[:, vec_idx] -= sum
        # normalise vector
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx] / norm(orth_matrix[:, vec_idx])
    end
    return orth_matrix
end # function

"""
    modified_gram_schmidt(matrix)

Apply the modified Gram-Schmidt orthogonalization process to the column vectors
of the input matrix `matrix`.

See also [classical_gram_schmidt](@ref), [modified_gram_schmidt_rutishauser](@ref)
"""
function modified_gram_schmidt(matrix)
    # orthogonalises the columns of the input matrix
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx] / norm(orth_matrix[:, vec_idx])
        for span_base_idx = (vec_idx + 1):num_vectors
            # perform block step
            orth_matrix[:, span_base_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx]) * orth_matrix[:, vec_idx]
        end
    end
    return orth_matrix
end # function

"""
modified_gram_schmidt_rutishauser(matrix)

Apply the modified Gram-Schmidt orthogonalization (the variant proposed by Rutishauser) 
process to the column vectors of the input matrix `matrix`.

See also [classical_gram_schmidt](@ref), [modified_gram_schmidt](@ref)
"""
function modified_gram_schmidt_rutishauser(matrix)
    # orthogonalises the columns of the input matrix
    num_vectors = size(matrix)[2]
    orth_matrix = copy(matrix)
    for vec_idx = 1:num_vectors
        for span_base_idx = 1:(vec_idx - 1)
            # substrack sum
            orth_matrix[:, vec_idx] -= dot(orth_matrix[:, span_base_idx], orth_matrix[:, vec_idx]) * orth_matrix[:, span_base_idx]
        end
        # normalise vector
        orth_matrix[:, vec_idx] = orth_matrix[:, vec_idx] / norm(orth_matrix[:, vec_idx])
    end
    return orth_matrix
end # function

end # module


module TestOthonormalization 

using Test 
using GramSchmidtOthonormalization 


v = [0.74 -0.55 0.4; 0.66 0.45 -0.6; 0.15 0.7 0.7]
ourMat = Matrix([0.5 0.15  0.35;0.45 0.45 0.1;0.1 0.3 0.6])

@test round.(classical_gram_schmidt(ourMat);digits=2) == v

end





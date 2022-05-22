from sympy import *
import sympy
G = zeros(4, 2)
G[2:, :] = eye(2)

sa = symbols('sa', real=True)
D = eye(2)*sa**2

GDGt = G@D@G.T

w = symbols('w', real=True)
T = symbols('T', real=True)
A1 = zeros(4, 4)
A1[:2, 2:] = eye(2)
A1[2, 3] = -w
A1[3, 2] = w 

pprint(A1)

#Z = sympy.exp(M)

Ad = sympy.exp(A1*T)
Ad = Ad.rewrite(exp).simplify()
Ad = [a.trigsimp() for a in Ad.rewrite(cos).expand().as_real_imag()][0]
#Ad = Ad.subs(w, 2*pi/T)

def pprint_m(A):
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            print(A[row, col], end="      ")
        print()

print("A discrete")
pprint_m(Ad)
print("\n\n")



## Generating Q with Van Loan 
M = zeros(8, 8)
M[:4, :4] = -A1
M[:4, 4:] = GDGt
M[4:, 4:] = A1.T
M = M*T
Z = simplify(sympy.exp(M))
V2 = Z[:4, 4:]
V1 = Z[4:, 4:]

Q = V1.T@V2
Q = simplify([a.trigsimp() for a in Q.rewrite(cos).expand().as_real_imag()][0])
print("Q before variable substitution")
pprint_m(Q)
print("\n\n")
Q = Q.subs(w, 2*pi/T)

print("Q after variable substitution")
pprint_m(Q)
print("\n\n")






## Observing term within integral in equation 4.60
tau = symbols('t')
exp_Attau = Matrix([[1, 0 , sin((T-tau)*w)/w, (-1+cos((T-tau)*w))/w], 
[0, 1, (1-cos((T-tau)*w))/w, sin((T-tau)*w)/w], 
[0, 0, cos((T-tau)*w), -sin((T-tau)*w)], 
[0, 0, sin((T-tau)*w), cos((T-tau)*w)]])


print("Term inside integral")
pprint_m(simplify(exp_Attau@GDGt@exp_Attau.T))


using Symbolics
# let's first define two variables x and y
@variables x y
# First, we can construct symbolic terms via evaluations. This is also
# known as tracing
x^2 + y^2
# Because we are differential equations people, of course, the most basic
# operation would be to differentiate a given symbolic expression.
Symbolics.derivative(x^2 + y^2, x) # R->R
Symbolics.jacobian(x^2 + y^2, [x, y]) # R^2 -> R
Symbolics.gradient(x^2 + y^2, [x, y])
Symbolics.jacobian([x^2 + y^2; sin(y)], [x, y]) # R^2 -> R^2
# Another basic symbolic manipulation would be the substitution
substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>y^2))
substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>1.0))
# Trig rules
simplify(sin(x)^2 + 2 + cos(x)^2)
# constructor level simplify
2x - x
ex = x^2 + y^2 + sin(x)
ex - ex
ex / ex
ex^2 / ex

# Sometimes, you might not want to symbolicly trace into a function.
# For instance, a function with randomness in it.
foo(x, y) = x * rand() + y
foo(x, y)
@register foo(x, y)
foo(x, y)
Symbolics.derivative(foo(hypot(x, y), y), x)

# By default, the symbolic variable has the type Real. An, Symbolics
# is also generic enough to construct Complex valued symbolic variables
# as well.
@variables z::Complex
z^2
real(z^2)

# We can also construct array valued symbolic variable
@variables xs[1:5]
sum(xs)
Symbolics.scalarize(sum(xs))
# array valued symbolics are still at its early stage
sum(collect(xs))

# Now, it's a good time to switch gear and 
# Google Rosenbrock function;
rosenbrock(xs) = sum(1:length(xs)-1) do i
    100*(xs[i+1] - xs[i]^2)^2 + (1 - xs[i])^2
end

N = 100
xs = ones(N)
rosenbrock(xs)

# Symbolic tracing
@variables xs[1:N]
xs = collect(xs)
rxs = rosenbrock(xs)
grad = Symbolics.gradient(rxs, xs)
hes = Symbolics.jacobian(grad, xs)
sparse_hes = Symbolics.sparsejacobian(grad, xs)
Symbolics.jacobian_sparsity(grad, xs)
#=
using Plots, SparseArrays
spy(sparse_hes) # typo
spy(.!iszero.(sparse_hes))
=#

using BenchmarkTools
@benchmark(Symbolics.jacobian($grad, $xs))
@benchmark(Symbolics.sparsejacobian($grad, $xs)) # much faster
@benchmark(Symbolics.jacobian_sparsity($grad, $xs)) # much faster

@benchmark(Symbolics.hessian($rxs, $xs))
@benchmark(Symbolics.sparsehessian($rxs, $xs)) # a little bit faster
@benchmark(Symbolics.hessian_sparsity($rxs, $xs)) # a little bit faster

foop, fip = build_function(grad, xs, expression=Val{false});
aa = rand(N);
out = similar(aa);
fip(out, aa)
foop(aa) ≈ out
using ForwardDiff # varify it via AD
out ≈ ForwardDiff.gradient(rosenbrock, aa)

# Jacobian
hoop, hip = build_function(sparse_hes, xs, expression=Val{false});
hoop(aa) ≈ ForwardDiff.hessian(rosenbrock, aa)


using OrdinaryDiffEq, IfElse
brusselator_f(x, y, t) = IfElse.ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) & (t >= 1.1), 5.0, 0.0)
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a
function brusselator_2d_loop(du, u, p, t)
    alpha, xyd = p
    dx = step(xyd)
    N = length(xyd)
    alpha = alpha/dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[I[1]], xyd[I[2]]
        ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)

        ii = 1
        la1 = alpha*(u[im1,j,ii] + u[ip1,j,ii] + u[i,jp1,ii] + u[i,jm1,ii] - 4u[i,j,ii])
        ii = 2
        la2 = alpha*(u[im1,j,ii] + u[ip1,j,ii] + u[i,jp1,ii] + u[i,jm1,ii] - 4u[i,j,ii])

        du[i,j,1] = la1 + 1 + u[i,j,1]^2*u[i,j,2] - 4.4*u[i,j,1] + brusselator_f(x, y, t)
        du[i,j,2] = la2 + 3.4*u[i,j,1] - u[i,j,1]^2*u[i,j,2]
    end
    return nothing
end
function init_brusselator_2d(xyd)
  N = length(xyd)
  u = zeros(N, N, 2)
  for I in CartesianIndices((N, N))
    x = xyd[I[1]]
    y = xyd[I[2]]
    u[I,1] = 22*(y*(1-y))^(3/2)
    u[I,2] = 27*(x*(1-x))^(3/2)
  end
  u
end
xyd_brusselator = range(0,stop=1,length=32)

prob_ode_brusselator_2d = ODEProblem(brusselator_2d_loop,
                                     init_brusselator_2d(xyd_brusselator),
                                     (0.0, 6.0),
                                     (0.1,
                                      xyd_brusselator,))
saveat = [0.0, 0.5, 1.0, 1.3, 1.4, 5.6, 6.0]
sol = solve(prob_ode_brusselator_2d, KenCarp4(), saveat=saveat)
@variables u[axes(prob_ode_brusselator_2d.u0)...] t
u = collect(u)
du = similar(u)
brusselator_2d_loop(du, u, prob_ode_brusselator_2d.p, t)
sparsity = Symbolics.jacobian_sparsity(du, u)
@time sol = solve(prob_ode_brusselator_2d, KenCarp4(), saveat=saveat);
@time sol_sparsity = solve(remake(prob_ode_brusselator_2d, f=ODEFunction{true}(brusselator_2d_loop, jac_prototype=float(sparsity))), KenCarp4(), saveat=saveat);
sol_sparsity.u[end] ≈ sol.u[end]

plotat(t) = begin
    surface(xyd_brusselator, xyd_brusselator, sol(t)[:, :, 1], zlims=(0, 5))
    surface!(xyd_brusselator, xyd_brusselator, sol(t)[:, :, 2], zlims=(0, 5))
end

plotat(1.4)
plotat(5.6)

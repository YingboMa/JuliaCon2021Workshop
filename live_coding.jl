using Symbolics, ModelingToolkit, Plots, DifferentialEquations

@variables x y
x^2 + y^2
Symbolics.derivative(x^2 + y^2, x)
Symbolics.gradient(x^2 + y^2, [x, y])
Symbolics.jacobian([x^2 + y^2; y^2], [x, y])

substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>y^2))
substitute(sin(x)^2 + 2 + cos(x)^2, Dict(x=>1.0))

simplify(sin(x)^2 + 2 + cos(x)^2)
foo(x, y) = x * rand() + y
@register foo(x, y)
Symbolics.derivative(foo(hypot(x, y), y), x)

@variables z::Complex

@variables xs[1:10]

rosenbrock(xs) = sum(1:length(xs)-1) do i
    100*(xs[i+1] - xs[i]^2)^2 + (1 - xs[i])^2
end

N = 100
xs = ones(N)
rosenbrock(xs)

@variables xs[1:N]
xs = collect(xs)
rxs = rosenbrock(xs)
grad = Symbolics.gradient(rxs, xs)
hes = Symbolics.jacobian(grad, xs) # Hessian is Jacobian of the gradient
hes_sparse = Symbolics.sparsejacobian(grad, xs) # Hessian is Jacobian of the gradient
hes_sparsity = Symbolics.jacobian_sparsity(grad, xs) # Hessian is Jacobian of the gradient

using Plots
spy(hes_sparse)
spy(hes_sparsity)

using BenchmarkTools
@benchmark Symbolics.jacobian($grad, $xs)
@benchmark Symbolics.sparsejacobian($grad, $xs) # Hessian is Jacobian of the gradient
@benchmark Symbolics.jacobian_sparsity($grad, $xs) # Hessian is Jacobian of the gradient


@benchmark Symbolics.hessian($rxs, $xs)
@benchmark Symbolics.sparsehessian($rxs, $xs) # Hessian is Jacobian of the gradient
@benchmark Symbolics.hessian_sparsity($rxs, $xs) # Hessian is Jacobian of the gradient

foop, fip = build_function(grad, xs)
foop, fip = build_function(grad, xs; expression = Val{false})
aa = rand(N)
out = similar(aa)
fip(out, aa)
foop(aa)
foop(aa) ≈ out
using ForwardDiff
ForwardDiff.gradient(rosenbrock, aa) ≈ out


hoop, hip = build_function(hes_sparse, xs; expression = Val{false})
hoop(aa)
hout = similar(hes_sparse, Float64)
hip(hout, aa)
ForwardDiff.hessian(rosenbrock, aa) ≈ hout



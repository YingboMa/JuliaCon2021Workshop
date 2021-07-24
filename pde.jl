using OrdinaryDiffEq, IfElse, Plots, Symbolics
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

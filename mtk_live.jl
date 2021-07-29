using ModelingToolkit, OrdinaryDiffEq

@parameters t
D = Differential(t)

@connector function Pin(;name)
    sts = @variables v(t)=1.0 i(t)=1.0
    ODESystem(Equation[], t, sts, [], name=name)
end

function ModelingToolkit.connect(::Type{Pin}, ps...)
    # KCL
    eqs = [0 ~ sum(p->p.i, ps)]
    # KVL
    for i in 1:length(ps)-1
        push!(eqs, ps[i].v ~ ps[i+1].v)
    end
    eqs
end

function Ground(;name)
    @named g = Pin()
    eqs = [0 ~ g.v]
    compose(ODESystem(eqs, t, [], [], name=name), g)
end

function OnePort(;name)
    @named p = Pin()
    @named n = Pin()
    sts = @variables v(t)=1.0 i(t)=1.0
    eqs = [
           v ~ p.v - n.v
           0 ~ p.i + n.i
           i ~ p.i
          ]
    compose(ODESystem(eqs, t, sts, []; name=name), p, n)
end

function Resistor(;name, R=1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    ps = @parameters R=R
    eqs = [v ~ i * R]
    extend(ODESystem(eqs, t, [], ps; name=name), oneport)
end

function Capacitor(;name, C=1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    ps = @parameters C=C
    eqs = [D(v) ~ i / C]
    extend(ODESystem(eqs, t, [], ps; name=name), oneport)
end

function ConstantVoltage(;name, V=1.0)
    @named oneport = OnePort()
    @unpack v = oneport
    ps = @parameters V=V
    eqs = [v ~ V]
    extend(ODESystem(eqs, t, [], ps; name=name), oneport)
end

R = 1.0
C = 1.0
V = 1.0
@named resistor = Resistor(R=R)
@named capacitor = Capacitor(C=C)
@named source = ConstantVoltage(V=V)
@named ground = Ground()

rc_eqs = [
          connect(source.p, resistor.p)
          connect(resistor.n, capacitor.p)
          connect(capacitor.n, source.n, ground.g)
         ]
@named rc_model = compose(ODESystem(rc_eqs, t), [resistor, capacitor, source, ground])

sys = structural_simplify(rc_model)
u0 = [capacitor.v => 0.0]
prob = ODAEProblem(sys, u0, (0.0, 10.0))
sol = solve(prob, Tsit5())
plot(sol)
plot(sol, vars = [capacitor.v, resistor.v])

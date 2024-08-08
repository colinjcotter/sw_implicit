from firedrake import *

mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh, "DG", 0)
W = V * V * V
# q_v, R_c, R_e

q_T = Constant(1.0)

U0 = Function(W)
q0_v, R0_c, R0_e = split(U0)

U1 = Function(W)
q1_v, R1_c, R1_e = split(U1)

tc = Constant(0.)

dt = 0.05
dtc = Constant(dt)

qsat = Constant(1.2) - sin(tc)

dq_v, dR_c, dR_e = TestFunctions(W)

eqn = (
    dq_v*(q1_v - q0_v
          + dtc*(R1_c - R1_e))
    + dR_c*(qsat - q1_v)
    + dR_e*(qsat - q1_v)*(q_T - q1_v -R1_e)
    )*dx

# variational solver
params = {
    "snes_type": "vinewtonrsls",
    #"snes_vi_monitor": None,
    #"snes_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

nproblem = NonlinearVariationalProblem(
    eqn, U1)
nsolver = NonlinearVariationalSolver(
    nproblem, solver_parameters=params)

# set the bounds

lbound = Function(W).assign(PETSc.NINFINITY)
ubound = Function(W).assign(PETSc.INFINITY)

lbound.sub(1).assign(0.) #  R_c >= 0

q0_v, R0_c, R0_e = U0.subfunctions
q0_v.assign(q_T)

T = 10.
t = 0.
U1.assign(U0)
while t < T - dt:
    nsolver.solve(bounds=(lbound, ubound))
    U0.assign(U1)
    print(t, q0_v.dat.data[:], float(qsat),
          R0_c.dat.data[:], R0_e.dat.data[:])
    t += dt
    tc.assign(t)

from firedrake import *

horizontal_degree = 0
vertical_degree = 0

L = 3.0e5
H = 1.0e4
nlayers = 50
mesh = ExtrudedMesh(PeriodicIntervalMesh(100, L), nlayers, H / nlayers)
S1family = "CG"

S1 = FiniteElement(S1family, mesh._base_mesh.ufl_cell(), horizontal_degree+1)
S2 = FiniteElement("DG", mesh._base_mesh.ufl_cell(), horizontal_degree)
T0 = FiniteElement("CG", interval, vertical_degree+1)
T1 = FiniteElement("DG", interval, vertical_degree)

Tlinear = FiniteElement("CG", interval, 1)
VT_elt =TensorProductElement(S2, Tlinear)

V2h_elt = HDiv(TensorProductElement(S1, T1))
V2t_elt = TensorProductElement(S2, T0)
V2v_elt = HDiv(V2t_elt)
V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
V3_elt = TensorProductElement(S2, T1)
V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)

V = FunctionSpace(mesh, V2_elt)
remapped = WithMapping(V2_elt, "identity")
V = FunctionSpace(mesh, remapped, name="HDiv")

T = FunctionSpace(mesh, VT_elt)

W = V * T


##################################################

u, lambdar = TrialFunctions(W)
v, gammar = TestFunctions(W)

n = FacetNormal(mesh)

a = (inner(u, v) * dx + div(u) *div(v) * dx
     + inner(v, n) * lambdar * (ds_b + ds_t)
     + inner(u, n) * gammar * (ds_b + ds_t)
     + jump(v, n=n) * lambdar('+') * (dS_h)
     + jump(u, n=n) * gammar('+') * (dS_h))

x = SpatialCoordinate(mesh)
if len(x) == 2:
    rsq = (x[0] - L/2) ** 2 / 20 ** 2 + (x[1] - H/2) ** 2 / 0.2 ** 2
else:
    rsq = (x[0] - L/2) ** 2 / 20 ** 2 + (x[1] - 50) ** 2 / 20 ** 2 + (x[2] - H/2) ** 2 / 0.2 ** 2
f = as_vector([exp(cos(2*pi*x[0]/L)*sin(pi*x[1]/H)), cos(cos(2*pi*x[0]/L)*sin(pi*x[1]/H))])
L = inner(f, v)*dx
    
wh = Function(W)
problem = LinearVariationalProblem(a, L, wh)

sparameters = {
    "mat_type": "matfree",
    'snes_monitor': None,
    "ksp_type": "gmres",
    "ksp_view": None,
    "ksp_monitor": None,
    'pc_type': 'python',
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_ksp_type": "preonly",
    "assembled_pc_type": "python",
    'assembled_pc_python_type': 'firedrake.ASMStarPC',
    'assembled_pc_star_dims': '0',
    "assembled_pc_star_sub_pc_type": "lu",
    'assembled_pc_star_sub_sub_pc_factor_mat_solver_type' : 'mumps',
}

LS_solver = LinearVariationalSolver(problem, solver_parameters=sparameters)
LS_solver.solve()

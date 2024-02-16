import firedrake as fd
from firedrake.petsc import PETSc
import argparse

from utils import units
from utils.planets import earth
from utils import shallow_water as swe
from utils.shallow_water.williamson1992 import case5

PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser(
    description='Williamson 5 testcase for augmented Lagrangian solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')  # noqa: E501
parser.add_argument('--dmax', type=float, default=15, help='Final time in days.')  # noqa: E501
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours.')  # noqa: E501
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours.')  # noqa: E501
parser.add_argument('--filename', type=str, default='w5hybr')  # noqa: E501
parser.add_argument('--write_file', action='store_true', help='Write time series to vtk file.')  # noqa: E501
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of mesh coordinates.')  # noqa: E501
parser.add_argument('--degree', type=int, default=1, help='Degree of the DG pressure finite element space.')  # noqa: E501
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')  # noqa: E501

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                refinement_level=args.ref_level,
                                degree=args.coords_degree)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

# function spaces
W = swe.default_function_space(mesh, degree=args.degree)
V1, V2 = W.subfunctions
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)  # potential vorticity space

# Trace space for hybridisation
V1b = fd.FunctionSpace(mesh, fd.BrokenElement(V1.ufl_element()))
Tr = fd.FunctionSpace(mesh, "HDivT", args.degree+1)
Wtr = V1b * V2 * Tr
Wb = V1b * V2

H = case5.H0
f = case5.coriolis_expression(*x)  # Coriolis parameter
g = earth.Gravity  # Gravitational constant
b = case5.topography_expression(*x)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.nonlinear.form_function(mesh, g, b, f, u, h, v, q, t)


def linear_form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f, u, h, v, q, t)


Un = fd.Function(W)
Un1 = fd.Function(W)

dt = units.hour*args.dt
dT = fd.Constant(dt)

u0s = fd.split(Un)
u1s = fd.split(Un1)
vs = fd.TestFunctions(W)

half = fd.Constant(0.5)

eqn = (
    form_mass(*u1s, *vs)
    - form_mass(*u0s, *vs)
    + half*dT*form_function(*u1s, *vs)
    + half*dT*form_function(*u0s, *vs)
)


# PC forming approximate hybridisable system (without advection)
# solve it using hybridisation and then return the DG part
# (for use in a Schur complement setup)
class ApproxHybridPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        # input and output functions
        self.xfstar = fd.Cofunction(V2.dual())
        self.xf = fd.Function(V2)  # result of riesz map of the above
        self.yf = fd.Function(V2)  # the preconditioned residual

        # hybridised system
        v, q, dll = fd.TestFunctions(Wtr)
        w0 = fd.Function(Wtr)
        u, p, ll = fd.TrialFunctions(Wtr)
        _, self.p0, _ = w0.subfunctions

        n = fd.FacetNormal(mesh)
        eqn = (
            form_mass(u, p, v, q)
            + half*dT*linear_form_function(u, p, v, q)
        )

        # trace bits
        eqn += (
            0.5*g*dT*fd.jump(v, n)*ll("+")
            + fd.jump(u, n)*dll("+")
        )*fd.dS

        # the rhs
        eqn -= q*self.xf*fd.dx

        factorisation_params = {
            'ksp_type': 'preonly',
            'pc_factor_mat_ordering_type': 'rcm',
            'pc_factor_reuse_ordering': None,
            'pc_factor_reuse_fill': None,
        }

        lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        lu_params.update(factorisation_params)

        ilu_params = {'pc_type': 'ilu'}
        ilu_params.update(factorisation_params)

        gamg_params = {
            # 'ksp_type': 'preonly',
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-10,
            "ksp_converged_reason": None,
            'pc_type': 'gamg',
            # 'pc_gamg_sym_graph': None,
            'pc_mg_type': 'full',
            'pc_mg_cycle_type': 'v',
            'mg': {
                'levels': {
                    'ksp_type': 'gmres',
                    'ksp_max_it': 5,
                    'pc_type': 'bjacobi',
                    'sub': ilu_params,
                },
                'coarse': lu_params
            }
        }

        hbps = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field': lu_params
            # "ksp_type": "gmres",
            # "ksp_rtol": 1e-5,
            # 'condensed_field': gamg_params
        }

        prob = fd.LinearVariationalProblem(fd.lhs(eqn), fd.rhs(eqn), w0,
                                           constant_jacobian=True)
        self.solver = fd.LinearVariationalSolver(
            prob, solver_parameters=hbps, options_prefix="hybr")

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        self.xf.assign(self.xfstar.riesz_representation())
        self.solver.solve()
        self.yf.assign(self.p0)

        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)


# PC forming approximate schur complement (without advection)
# of the broken system, which is block diagonal.
# (for use in a Schur complement setup)
class ApproxSchurPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        # input and output functions
        self.xfstar = fd.Cofunction(V2.dual())
        self.xf = fd.Function(V2)  # result of riesz map of the above
        self.yf = fd.Function(V2)  # the preconditioned residual

        # hybridised system
        v, q = fd.TestFunctions(Wb)
        w0 = fd.Function(Wb)
        u, p = fd.TrialFunctions(Wb)
        _, self.p0 = w0.subfunctions

        n = fd.FacetNormal(mesh)
        eqn = (
            form_mass(u, p, v, q)
            + half*dT*linear_form_function(u, p, v, q)
        )

        # the rhs
        eqn -= q*self.xf*fd.dx

        factorisation_params = {
            'ksp_type': 'preonly',
            'pc_factor_mat_ordering_type': 'rcm',
            'pc_factor_reuse_ordering': None,
            'pc_factor_reuse_fill': None,
        }

        lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        lu_params.update(factorisation_params)

        ilu_params = {'pc_type': 'ilu'}
        ilu_params.update(factorisation_params)

        schur_params = {
            'ksp_type': 'preonly',
            # 'ksp_monitor': None,
            # 'ksp_converged_reason': None,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_schur_precondition': 'full',
            'fieldsplit': lu_params
        }

        prob = fd.LinearVariationalProblem(fd.lhs(eqn), fd.rhs(eqn), w0,
                                           constant_jacobian=True)
        self.solver = fd.LinearVariationalSolver(
            prob, solver_parameters=schur_params,
            options_prefix="schur")

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        self.xf.assign(self.xfstar.riesz_representation())
        self.solver.solve()
        self.yf.assign(self.p0)

        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)


atol = 1e4
sparameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "ksp_ew": None,
        "atol": atol,
    },
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
    },
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    "fieldsplit_1": {
        "ksp_type": "gmres",
        # "ksp_monitor": None,
        # "ksp_converged_reason": None,
        "ksp_rtol": 1e-3,
        "pc_type": "python",
        "pc_python_type": __name__ + ".ApproxSchurPC",
    }
}

nprob = fd.NonlinearVariationalProblem(eqn, Un1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters)

tmax = earth.day*args.dmax
dumpt = units.hour*args.dumpt
tdump = 0.

un = fd.Function(V1, name="Velocity").project(case5.velocity_expression(*x))
etan = fd.Function(V2, name="Elevation").project(case5.elevation_expression(*x))  # noqa: E501

# Topography.
u0, h0 = Un.subfunctions
u0.assign(un)
h0.project(etan + H - b)

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*fd.dx + fd.inner(fd.cross(fd.CellNormal(mesh), fd.grad(p)), un)*fd.dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type': 'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(args.filename+'.pvd')
etan.project(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Un1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0
t = 0.
while t < tmax + 0.5*dt:
    PETSc.Sys.Print("")
    PETSc.Sys.Print("=== --- === --- === --- === --- ===")
    PETSc.Sys.Print("")
    PETSc.Sys.Print(f">>> Calculating timestep {stepcount} at {round(t/60**2, 2)} hours")  # noqa: E501
    PETSc.Sys.Print("")
    t += dt

    nsolver.solve()
    Un.assign(Un1)

    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()

    if args.write_file:
        tdump += dt
        if tdump > dumpt - dt*0.5:
            etan.project(h0 - H + b)
            un.assign(u0)
            qsolver.solve()
            file_sw.write(un, etan, qn)
            tdump -= dumpt

PETSc.Sys.Print("")
PETSc.Sys.Print("Iterations", itcount,
                "its per step", round(itcount/stepcount, 2),
                "dt", dt/60**2,
                "ref_level", args.ref_level,
                "dmax", args.dmax)

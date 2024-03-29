from firedrake import *

import argparse
parser = argparse.ArgumentParser(description='Grad-Div Helmholtz solver in the plane.')
parser.add_argument('--base_level', type=int, default=3, help='Base refinement level of square grid for MG solve. Default 3.')
parser.add_argument('--ref_level', type=int, default=4, help='Refinement level of square grid. Default 4.')
parser.add_argument('--gamma', type=float, default=1.0e5, help='Augmented Lagrangian scaling parameter. Default 10000.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
args = parser.parse_known_args()
args = args[0]

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
#distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.FACET, 2)}

nbase = args.base_level
nrefs = args.ref_level
basemesh = UnitSquareMesh(nbase, nbase)
mh = MeshHierarchy(basemesh, nrefs)
mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", 2)

v = TestFunction(V)
u = TrialFunction(V)

gamma = Constant(args.gamma)
x, y = SpatialCoordinate(mesh)
f = exp(- ((x-0.5)**2 + (y-0.5)**2)/0.2**2) + conditional(y<0.5, 0, 1) 
a = inner(u, v)*dx + gamma*div(u)*div(v)*dx
L = v[0]*f*dx

kspmg = 3
sp = {
    "ksp_type": "gcr",
    "ksp_monitor": None,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_dense_inverse": True,
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_sub_pc_factor_mat_solver_type": "petsc",
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

u0 = Function(V)
myprob = LinearVariationalProblem(a, L, u0)
mysolver = LinearVariationalSolver(myprob,
                                   solver_parameters = sp)
mysolver.solve()

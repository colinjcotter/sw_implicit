from firedrake import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import mg
R0 = 1
base_level = 1
ref_level = 5
nrefs = ref_level - base_level
deg = 1
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
#distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.FACET, 2)}


def high_order_mesh_hierarchy(mh, degree, R0):
    meshes = []
    for m in mh:
        X = VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = interpolate(m.coordinates, X)
        new_mesh = Mesh(new_coords)
        meshes.append(new_mesh)

    return HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)

basemesh = IcosahedralSphereMesh(radius=R0,
                                 refinement_level=base_level,
                                 degree=1,
                                 distribution_parameters = distribution_parameters)
del basemesh._radius
mh = MeshHierarchy(basemesh, nrefs)
# creates higher order mesh hierarchy on icosahedron
# without pushing coordinates out the the sphere yet
mh = high_order_mesh_hierarchy(mh, deg, R0)
mesh = mh[-1]

# store the original coordinates on the mesh, then
# push the coordinates out to the sphere and orient the cells
for mesh in mh:
    xf = mesh.coordinates
    mesh.transfer_coordinates = Function(xf)
    x = SpatialCoordinate(mesh)
    r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    xf.interpolate(R0*xf/r)
    mesh.init_cell_orientations(x)
mesh = mh[-1]

R0 = Constant(R0)
V = FunctionSpace(mesh, "BDM", 2)

v = TestFunction(V)
u = TrialFunction(V)

gamma = Constant(1.0e5)
x, y, z = SpatialCoordinate(mesh)
f0 = Function(V)
import numpy as np
f0.dat.data[:] = np.random.randn(*(f0.dat.data[:].shape))
f = exp(- (x**2 + z**2)/0.2**2) + conditional(z<0, 0, 1) 
a = inner(u, v)*dx + gamma*div(u)*div(v)*dx
# L = v[0]*f*dx
L = inner(v, f0)*dx
kspmg = 3
sp = {
    "ksp_type": "fgmres",
    "ksp_view": None,
    "ksp_monitor": None,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
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

sp_patch = {
    "ksp_type": "cg",
    "ksp_view": None,
    "ksp_monitor": None,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_sub_mat_type": "seqaij",
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_multiplicative": False,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_dense_inverse": True,
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_sub_pc_factor_mat_solver_type": "petsc",
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
}

u0 = Function(V)
myprob = LinearVariationalProblem(a, L, u0)
mysolver = LinearVariationalSolver(myprob,
                                   solver_parameters = sp)
# passing in the SW coefficients, all 1
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
U = Function(W)
ubar, hbar = U.split()
ubar.assign(0.)
hbar.assign(1.0)
vtransfer = mg.ManifoldTransfer()
tm = TransferManager()
transfers = {
    V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, tm.inject)
}
transfermanager = TransferManager(native_transfers=transfers)
mysolver.set_transfer_manager(transfermanager)

for level in range(0):
    Ec = FunctionSpace(mh[-2-level], "CG", 3)
    Vc = FunctionSpace(mh[-2-level], "BDM", 2)
    Vf = FunctionSpace(mh[-1-level], "BDM", 2)
    psi = Function(Ec)
    psi.dat.data[:] = np.random.randn(*(psi.dat.data[:].shape))
    
    outward_normals = CellNormal(mh[-2-level])

    def perp(u):
        return cross(outward_normals, u)

    wc = Function(Vc)
    bkup_coords = Function(mh[-2-level].coordinates)
    mh[-2].coordinates.assign(mh[-2-level].transfer_coordinates)
    wc.project(perp(grad(psi)))
    mh[-2].coordinates.assign(bkup_coords)
    
    wf = Function(Vf)
    #vtransfer.prolong(wc, wf)
    tm.prolong(wc, wf)
    
    print(norm(wc), norm(wf), norm(div(wc)), norm(div(wf)))

mysolver.solve()

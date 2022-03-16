from firedrake import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import mg
R0 = 1
base_level = 0
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
#creates higher order mesh hierarchy on icosahedron
mh = high_order_mesh_hierarchy(mh, deg, R0)
mesh = mh[-1]
# store the mesh coordinates on the icosahedron before we overwrite them
transfer_coordinates = Function(mh[0].coordinates)

# push the coordinates out to the sphere and orient the cells
for mesh in mh:
    x = SpatialCoordinate(mesh)
    #r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    #xf = mesh.coordinates
    #xf.interpolate(R0*xf/r)
    mesh.init_cell_orientations(x)
mesh = mh[-1]

R0 = Constant(R0)
V = FunctionSpace(mesh, "BDM", 2)

v = TestFunction(V)
u = TrialFunction(V)

gamma = Constant(1.0e5)
x, y, z = SpatialCoordinate(mesh)
f = exp(- (x**2 + z**2)/0.2**2) + conditional(z<0, 0, 1) 
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
    "mg_levels_patch_pc_patch_construct_type": "vanka",
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
# passing in the SW coefficients, all 1
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
U = Function(W)
ubar, hbar = U.split()
ubar.assign(0.)
hbar.assign(1.0)
vtransfer = mg.SWTransfer(U, upwind=False, transfer_coordinates=False)
tm = TransferManager()
transfers = {
    V.ufl_element(): (vtransfer.prolong, tm.restrict, tm.inject)
}
transfermanager = TransferManager(native_transfers=transfers)
mysolver.set_transfer_manager(transfermanager)

mysolver.solve()


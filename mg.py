from firedrake import *
import numpy as np

def both(e):
    return e('+')+e('-')

class SWTransfer(object):
    def __init__(self, w0, upwind=True):
        '''
        Object to manage transfer operators for MG
        applied to augmented Lagrangian solver
        for implicit rotating shallow water equations.
        Assumes BDM spaces.

        :arg w0: A Firedrake function containing the 
        current value of the state of the nonlinear solver
        :arg upwind: True for using upwind values of delta h using ubar
        otherwise use the average.
        '''
        self.w0 = w0
        self.ubar, self.hbar = w0.split()
        mesh = self.ubar.ufl_domain()
        self.V = FunctionSpace(mesh,
                               self.ubar.function_space().ufl_element())
        self.degree = self.ubar.function_space().ufl_element().degree()
        self.upwind = upwind
        
        # list of flags to say if we've set up before
        self.ready = {}
        # list of coarse and fine fluxes
        self.w_coarse_b = {} #broken mixed variable for F coarse
        self.F_coarse_b = {} #pointer to the F part of the above
        self.F_fine_b = {} #broken variable for F fine
        self.coarse_weight = {} #averaging weight
        self.F_coarse = {}
        self.F_coarse_DG = {}
        self.F_fine_DG = {}
        self.F_fine = {}
        self.u_coarse = {}
        self.w_fine_b = {} #broken mixed variable for u fine
        self.u_fine_b = {} #pointer to the u part of the above
        self.fine_weight = {} #averaging weight
        self.ubar_coarse = {}
        self.ubar_fine = {}
        self.hbar_coarse = {}
        self.hbar_fine = {}
        self.coarse_solver = {}
        self.fine_solver = {}
        #self.Ftransfer = TransferManager(use_averaging=False)
        self.Ftransfer = TransferManager()
        self.coarse_average_kernel = {}
        self.fine_average_kernel = {}
        
    def prolong(self, coarse, fine):
        Vfine = FunctionSpace(fine.ufl_domain(),
                              fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True
            coarse_mesh = coarse.ufl_domain()
            coarse_element = coarse.function_space().ufl_element()
            Vcoarse = FunctionSpace(coarse_mesh, coarse_element)
            Vcoarseb = FunctionSpace(coarse_mesh,
                                     BrokenElement(coarse_element))
            degree = self.degree
            VcoarseDG = VectorFunctionSpace(coarse_mesh, "DG", degree)
            
            # make a solver du -> F (on coarse mesh)
            bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                               degree, variant="integral"))
            Ucoarse = FunctionSpace(coarse_mesh, bNed)
            Wcoarse = Vcoarseb * Ucoarse

            Qcoarse = FunctionSpace(coarse_mesh, "DG", degree-1)

            self.hbar_coarse[key] = Function(Qcoarse)
            self.ubar_coarse[key] = Function(Vcoarse)
            hbar_coarse = self.hbar_coarse[key]
            ubar_coarse = self.ubar_coarse[key]
            self.w_coarse_b[key] = Function(Wcoarse)
            self.F_coarse_b[key], _ = self.w_coarse_b[key].split()
            self.F_coarse_DG[key] = Function(VcoarseDG)
            self.F_coarse[key] = Function(Vcoarse)
            self.u_coarse[key] = Function(Vcoarse)
            n = FacetNormal(coarse_mesh)
            if self.upwind:
                Upwind = Constant(0.5) * (sign(dot(ubar_coarse, n)) + 1)
                hup = Upwind('+')*hbar_coarse('+') + \
                    Upwind('-')*hbar_coarse('-')
            else:
                hup = Constant(0.5)*(hbar_coarse('+') + hbar_coarse('-'))

            w, r = TestFunctions(Wcoarse)
            F, v = TrialFunctions(Wcoarse)
            
            a = both(inner(w, n)*inner(F, n))*dS
            a += inner(w, n)*inner(F, n)*ds
            a += inner(w, v)*dx
            a += inner(F, r)*dx

            L = both(inner(w, n)*inner(self.u_coarse[key], n))*hup*dS
            L += inner(w, n)*inner(self.u_coarse[key]*hbar_coarse, n)*ds
            L += inner(self.u_coarse[key]*hbar_coarse, r)*dx

            solver_parameters={
                "mat_type":"aij",
                "ksp_type":"preonly",
                "pc_type":"lu",
                "pc_factor_mat_solver_type" : "mumps"}

            coarse_prob = LinearVariationalProblem(a, L, self.w_coarse_b[key],
                                                   constant_jacobian=False)
            coarse_solver = LinearVariationalSolver(coarse_prob,
                                                    solver_parameters=
                                                    solver_parameters)
            self.coarse_solver[key] = coarse_solver

            shapes = (Vcoarse.finat_element.space_dimension(),
                      np.prod(Vcoarse.shape))
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            instructions = """
            for i, j
            w[i,j] = w[i,j] + 1
            end
            """
            self.coarse_weight[key] = Function(Vcoarse)
            weight = self.coarse_weight[key]
            par_loop((domain, instructions), dx, {"w": (weight, INC)},
                     is_loopy_kernel=True)
            instructions = """
            for i, j
            vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
            end
            """
            self.coarse_average_kernel[key] = (domain, instructions)

            # make a solver F -> du (on fine mesh)
            fine_mesh = fine.ufl_domain()
            fine_element = fine.function_space().ufl_element()
            Vfine = FunctionSpace(fine_mesh, fine_element)
            Vfineb = FunctionSpace(fine_mesh,
                                     BrokenElement(fine_element))
            degree = self.degree
            VfineDG = VectorFunctionSpace(fine_mesh, "DG", degree)

            bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                               degree, variant="integral"))
            Ufine = FunctionSpace(fine_mesh, bNed)
            Wfine = Vfineb * Ufine

            Qfine = FunctionSpace(fine_mesh, "DG", degree-1)

            self.hbar_fine[key] = Function(Qfine)
            self.ubar_fine[key] = Function(Vfine)
            self.w_fine_b[key] = Function(Wfine)
            self.u_fine_b[key], _ = self.w_fine_b[key].split()
            self.F_fine[key] = Function(Vfine)
            self.F_fine_b[key] = Function(Vfineb)
            self.F_fine_DG[key] = Function(VfineDG)
            hbar_fine = self.hbar_fine[key]
            ubar_fine = self.ubar_fine[key]
            n = FacetNormal(fine_mesh)
            if self.upwind:
                Upwind = Constant(0.5) * (sign(dot(ubar_fine, n)) + 1)
                hup = Upwind('+')*hbar_fine('+') + \
                    Upwind('-')*hbar_fine('-')
            else:
                hup = Constant(0.5)*(hbar_fine('+') + hbar_fine('-'))

            w, r = TestFunctions(Wfine)
            u, v = TrialFunctions(Wfine)

            a = both(inner(w, n)*inner(u, n))*hup*dS
            a += inner(w, n)*inner(u, n)*hbar_fine*ds
            a += inner(w, v)*hbar_fine*dx
            a += inner(u, r)*hbar_fine*dx

            F0 = self.F_fine[key]
            L = both(inner(w, n)*inner(F0, n))*dS
            L += inner(w, n)*inner(F0, n)*ds
            L += inner(F0, r)*dx
            
            fine_prob = LinearVariationalProblem(a, L, self.w_fine_b[key],
                                                   constant_jacobian=False)
            fine_solver = LinearVariationalSolver(fine_prob,
                                                  solver_parameters=
                                                  solver_parameters)
            self.fine_solver[key] = fine_solver

            shapes = (Vfine.finat_element.space_dimension(),
                      np.prod(Vfine.shape))
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            instructions = """
            for i, j
            w[i,j] = w[i,j] + 1
            end
            """
            self.fine_weight[key] = Function(Vfine)
            weight = self.fine_weight[key]
            par_loop((domain, instructions), dx, {"w": (weight, INC)},
                     is_loopy_kernel=True)
            instructions = """
            for i, j
            vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
            end
            """
            self.fine_average_kernel[key] = (domain, instructions)

            if hasattr(coarse_mesh, "transfer_coordinates"):
                if not hasattr(coarse_mesh, "coordinates_bk"):
                    coarse_mesh.coordinates_bk = Function(coarse_mesh.coordinates)
                if not hasattr(fine_mesh, "coordinates_bk"):
                    fine_mesh.coordinates_bk = Function(fine_mesh.coordinates)
                
        # update ubar and hbar on the levels
        keymax = max(list(self.ubar_fine.keys()))
        if keymax > key:
            self.Ftransfer.inject(self.ubar, self.ubar_fine[key])
        else:
            self.ubar_fine[key].assign(self.ubar)
        self.Ftransfer.inject(self.ubar, self.ubar_coarse[key])
        if keymax > key:
            self.Ftransfer.inject(self.hbar, self.hbar_fine[key])
        else:
            self.hbar_fine[key].assign(self.hbar)
        self.Ftransfer.inject(self.hbar, self.hbar_coarse[key])

        # copy coarse into the input to the coarse solver
        self.u_coarse[key].assign(coarse)

        # coarse solver produces w_coarse_b
        # This should be replaced with Slate
        #self.coarse_solver[key].solve()

        print("fix me")
        self.F_coarse[key].assign(coarse)

        #move the mesh
        if hasattr(self.F_coarse[key].ufl_domain(), "transfer_coordinates"):
            # change to the transfer coordinates for prolongation
            # e.g. if the mesh hierarchy is deformed to sphere
            self.F_coarse[key].ufl_domain().coordinates.assign(
                self.F_coarse[key].ufl_domain().transfer_coordinates)
            self.F_fine[key].ufl_domain().coordinates.assign(
                self.F_fine[key].ufl_domain().transfer_coordinates)

        # project F_coarse_b to a vector DG representation
        # projection solve is block diagonal but we should make a solver here
        # for speed
        self.F_coarse_DG[key].project(self.F_coarse_b[key])
        self.F_coarse_DG[key].project(self.F_coarse[key])
        print("fix me a")
        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.prolong(self.F_coarse_DG[key],
                               self.F_fine_DG[key])
        # project F_fine_DG into F_fine_b on flat mesh
        # projection solve is block diagonal but we should make a solver here
        self.F_fine_b[key].project(self.F_fine_DG[key])
        # average F_fine_b into F_fine
        self.F_fine[key].assign(0.)
        par_loop(self.fine_average_kernel[key], dx,
                 {"w": (self.fine_weight[key], READ),
                  "vec_in": (self.F_fine_b[key], READ),
                  "vec_out": (self.F_fine[key], INC)},
                is_loopy_kernel=True)

        #move the mesh back
        if hasattr(self.F_coarse[key].ufl_domain(), "transfer_coordinates"):
            #change back to deformed mesh
            self.F_coarse[key].ufl_domain().coordinates.assign(
                self.F_coarse[key].ufl_domain().coordinates_bk)
            self.F_fine[key].ufl_domain().coordinates.assign(
                self.F_fine[key].ufl_domain().coordinates_bk)

        # fine solver produces w_fine_b from F_fine
        print("repair me")
        fine.assign(self.F_fine[key])
        #self.fine_solver[key].solve()
        #fine.assign(0.)

        # average u_fine_b (split from w_fine_b) into fine
        #par_loop(self.fine_average_kernel[key], dx,
        #         {"w": (self.fine_weight[key], READ),
        #          "vec_in": (self.u_fine_b[key], READ),
        #          "vec_out": (fine, INC)},
        #        is_loopy_kernel=True)


class ManifoldTransfer(object):
    def __init__(self):
        '''
        Object to manage transfer operators for MG
        where we pull back to a piecewise flat mesh
        before doing transfers
        '''
        
        # list of flags to say if we've set up before
        self.ready = {}
        self.Ftransfer = TransferManager()

    def prolong(self, coarse, fine):
        Vfine = FunctionSpace(fine.ufl_domain(),
                              fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.prolong(coarse, fine)

        #change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)


    def restrict(self, fine, coarse):
        Vfine = FunctionSpace(fine.ufl_domain(),
                              fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.restrict(fine, coarse)

        #change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)

    def inject(self, fine, coarse):
        Vfine = FunctionSpace(fine.ufl_domain(),
                              fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.inject(fine, coarse)

        #change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)

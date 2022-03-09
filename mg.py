from firedrake import *

class SWTransfer(object):
    def __init__(self, w0):
        '''
        Object to manage transfer operators for MG
        applied to augmented Lagrangian solver
        for implicit rotating shallow water equations.
        Assumes BDM spaces.

        :arg w0: A Firedrake function containing the 
        current value of the state of the 
        '''
        self.w0 = w0
        self.ubar, self.hbar = w0.split()
        mesh = self.ubar.ufl_domain()
        self.V = FunctionSpace(mesh,
                               self.ubar.function_space().ufl_element())
        self.degree = self.ubar.function_space().ufl_element().degree()

        # list of flags to say if we've set up before
        self.ready = {}
        # list of coarse and fine fluxes
        self.F_coarse = {}
        self.F_fine = {}
        self.u_coarse = {}
        self.u_fine = {}
        self.ubar_coarse = {}
        self.ubar_fine = {}
        self.hbar_coarse = {}
        self.hbar_fine = {}
        self.coarse_solver = {}
        self.fine_solver = {}
        self.Ftransfer = TransferManager()
        
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
            degree = self.degree
            
            # make a solver du -> F (on coarse mesh)
            Tr = FunctionSpace(coarse_mesh, "HDiv Trace", degree)
            bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                               degree-1, variant="integral"))
            U = FunctionSpace(coarse_mesh, bNed)
            Wcoarse = Tr * U

            Qcoarse = FunctionSpace(coarse_mesh, "DG", degree-1)

            gamma, w = TestFunctions(Wcoarse)
            F = TrialFunction(Vcoarse)

            self.hbar_coarse[key] = Function(Qcoarse)
            self.ubar_coarse[key] = Function(Vcoarse)
            hbar_coarse = self.hbar_coarse[key]
            ubar_coarse = self.ubar_coarse[key]
            self.F_coarse[key] = Function(Vcoarse)
            self.u_coarse[key] = Function(Vcoarse)
            n = FacetNormal(coarse_mesh)
            Upwind = 0.5 * (sign(dot(ubar_coarse, n)) + 1)
            hup = Upwind('+')*hbar_coarse('+') + \
                Upwind('-')*hbar_coarse('-')

            a = gamma('+')*inner(F('+'), n('+'))*dS
            a += inner(w, F)*dx
            L = gamma('+')*inner(self.u_coarse[key]('+'), n('+'))*hup*dS
            L += inner(hbar_coarse*w, self.u_coarse[key])*dx

            solver_parameters={
                "mat_type":"aij",
                "ksp_type":"preonly",
                "pc_type":"lu",
                "pc_factor_mat_solver_type" : "mumps"}

            coarse_prob = LinearVariationalProblem(a, L, self.F_coarse[key],
                                                   constant_jacobian=False)
            coarse_solver = LinearVariationalSolver(coarse_prob,
                                                    solver_parameters=
                                                    solver_parameters)
            self.coarse_solver[key] = coarse_solver

            #make a solver F -> du (on fine mesh)
            fine_mesh = fine.ufl_domain()
            fine_element = fine.function_space().ufl_element()
            Vfine = FunctionSpace(fine_mesh, fine_element)
            degree = self.degree

            Tr = FunctionSpace(fine_mesh, "HDiv Trace", degree)
            bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                               degree-1, variant="integral"))
            U = FunctionSpace(fine_mesh, bNed)
            Wfine = Tr * U

            Qfine = FunctionSpace(fine_mesh, "DG", degree-1)

            gamma, w = TestFunctions(Wfine)
            F = TrialFunction(Vfine)

            self.hbar_fine[key] = Function(Qfine)
            self.ubar_fine[key] = Function(Vfine)
            self.F_fine[key] = Function(Vfine)
            self.u_fine[key] = Function(Vfine)
            hbar_fine = self.hbar_fine[key]
            ubar_fine = self.ubar_fine[key]
            n = FacetNormal(fine_mesh)
            Upwind = 0.5 * (sign(dot(ubar_fine, n)) + 1)
            hup = Upwind('+')*hbar_fine('+') + \
                Upwind('-')*hbar_fine('-')

            gamma, w = TestFunctions(Wfine)
            u = TrialFunction(Vfine)
            a = gamma('+')*inner(u('+'), n('+'))*hup*dS
            a += inner(hbar_fine*w, u)*dx
            L = gamma('+')*inner(self.F_fine[key]('+'), n('+'))*dS
            L += inner(w, self.F_fine[key])*dx

            fine_prob = LinearVariationalProblem(a, L, self.u_fine[key],
                                                   constant_jacobian=False)
            fine_solver = LinearVariationalSolver(fine_prob,
                                                  solver_parameters=
                                                  solver_parameters)
            self.fine_solver[key] = fine_solver

        # update ubar and ubar on the levels
        self.Ftransfer.inject(self.ubar, self.ubar_fine[key])
        self.Ftransfer.inject(self.ubar, self.ubar_coarse[key])
        # copy coarse into the input to the coarse solver
        self.u_coarse[key].assign(coarse)
        # coarse solver produces F_coarse
        self.coarse_solver[key].solve()
        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.prolong(self.F_coarse[key], self.F_fine[key])
        # fine solver produces u_fine from F_fine
        self.fine_solver[key].solve()
        # copy u_fine into fine
        fine.assign(self.u_fine[key])

import firedrake as fd
import numpy as np

def pi_formula(rho, theta, R_d, p_0, kappa):
    return (rho * R_d * theta / p_0) ** (kappa / (1 - kappa))

def hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary,
                    cp, R_d, p_0, kappa, g, Up,
                    top = False, Pi = None):
    # Calculate hydrostatic Pi, rho
    W_h = Vv * V2

    n = fd.FacetNormal(mesh)
    
    wh = fd.Function(W_h)
    v, rho = wh.split()
    rho.assign(rhon)
    v, rho = fd.split(wh)
    dv, drho = fd.TestFunctions(W_h)

    Pif = pi_formula(rho, thetan, R_d, p_0, kappa)

    rhoeqn = (
        (cp*fd.inner(v, dv) - cp*fd.div(dv*thetan)*Pif)*fd.dx
        + drho*fd.div(thetan*v)*fd.dx
    )
    
    if top:
        bmeasure = fd.ds_t
        bstring = "bottom"
    else:
        bmeasure = fd.ds_b
        bstring = "top"

    rhoeqn += cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
    rhoeqn += g*fd.inner(dv, Up)*fd.dx
    bcs = [fd.DirichletBC(W_h.sub(0), fd.as_vector([fd.Constant(0.0),
                                                    fd.Constant(0.0)]), bstring)]

    RhoProblem = fd.NonlinearVariationalProblem(rhoeqn, wh, bcs=bcs)

    schur_params = {'ksp_type': 'gmres',
                    'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'pc_fieldsplit_schur_fact_type': 'full',
                    'pc_fieldsplit_schur_precondition': 'selfp',
                    'fieldsplit_1_ksp_type': 'preonly',
                    'fieldsplit_1_pc_type': 'gamg',
                    'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                    'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                    'fieldsplit_0_ksp_type': 'richardson',
                    'fieldsplit_0_ksp_max_it': 4,
                    'ksp_atol': 1.e-08,
                    'ksp_rtol': 1.e-08}

    RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                              solver_parameters=schur_params,
                                              options_prefix="rhosolver")

    RhoSolver.solve()
    v, Rho0 = wh.split()

    rhon.assign(Rho0)

    if Pi:
        Pi.project(pi_formula(rhon, thetan, R_d, p_0, kappa))

def theta_tendency(q, u, theta, n):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        q*fd.inner(u,fd.grad(theta))*fd.dx
        + fd.jump(q)*(unn('+')*theta('+')
                      - unn('-')*theta('-'))*fd.dS_v
        - fd.jump(q*u*theta, n)*fd.dS_v
    )

def theta_eqn(q, n, thetan, thetanp1, un, unp1, Up, dT):
    thetanph = 0.5*(thetan + thetanp1)
    unph = 0.5*(un + unp1)
    qsupg = q + fd.Constant(0.5)*dT*fd.inner(unph, Up)*fd.inner(fd.grad(q), Up)
    return (
        qsupg*(thetanp1 - thetan)*fd.dx
        + dT*theta_tendency(qsupg, unph, thetanph, n)
    )

def rho_tendency(q, rho, u, n):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        - fd.inner(fd.grad(q), u*rho)*fd.dx +
        fd.jump(q)*(unn('+')*rho('+')
                    - unn('-')*rho('-'))*(fd.dS_v + fd.dS_h)
    )

def rho_eqn(q, n, rhon, rhonp1, un, unp1, dT):
    unph = 0.5*(un + unp1)
    rhonph = 0.5*(rhon + rhonp1)
    
    return (
        q*(rhonp1 - rhon)*fd.dx
        + dT*rho_tendency(q, rhonph, unph, n)
    )

def curl0(u):
    """
    Curl function from y-cpt field to x-z field
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())
    
    if d == 2:
        # equivalent vector is (0, u, 0)

        # |i   j   k  |
        # |d_x 0   d_z| = (- du/dz, 0, du/dx)
        # |0   u   0  |
        return fd.as_vector([-u.dx(1), u.dx(0)])
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def curl1(u):
    """
    dual curl function from dim-1 forms to dim-2 forms
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # we have vector in x-z plane and return scalar
        # representing y component of the curl

        # |i   j   k   |
        # |d_x 0   d_z | = (0, -du_3/dx + du_1/dz, 0)
        # |u_1 0   u_3 |
        
        return -u[1].dx(0) + u[0].dx(1)
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def cross1(u, w):
    """
    cross product (slice vector field with slice vector field)
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt

        # |i   j   k   |
        # |u_1 0   u_3 | = (0, -u_1*w_3 + u_3*w_1, 0)
        # |w_1 0   w_3 |

        return w[0]*u[1] - w[1]*u[0]
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """

    # |i   j   k   |
    # |u_1 0   u_3 | = (-w*u_3, 0, w*u_1)
    # |0   w   0   |

    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return fd.as_vector([-w*u[1], w*u[0]])
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError
    

def both(u):
    return 2*fd.avg(u)

    
def u_eqn(w, n, un, unp1, thetan, thetanp1, rhon, rhonp1,
          cp, g, R_d, p_0, kappa, Up, dT, mu=None):
    """
    Written in a dimension agnostic way
    """
    unph = 0.5*(un + unp1)
    thetanph = 0.5*(thetan + thetanp1)
    rhonph = 0.5*(rhon + rhonp1)
    Pinph = pi_formula(rhonph, thetanph, R_d, p_0, kappa)
    
    K = fd.Constant(0.5)*fd.inner(unph, unph)
    Upwind = 0.5*(fd.sign(fd.dot(unph, n))+1)

    eqn = (
        fd.inner(w, unp1 - un)*fd.dx
        + dT*fd.inner(unph, curl0(cross1(unph, w)))*fd.dx
        - dT*fd.inner(both(Upwind*unph),
                      both(cross0(n, cross1(unph, w))))*(fd.dS_h + fd.dS_v)
        - dT*fd.div(w)*K*fd.dx
        - dT*cp*fd.div(thetanph*w)*Pinph*fd.dx
        + dT*cp*fd.jump(w*thetanph, n)*fd.avg(Pinph)*fd.dS_v
        + dT*fd.inner(w, Up)*g*fd.dx
        )

    if mu:
        eqn += mu*fd.inner(w, Up)*fd.inner(unp1, Up)*fd.dx
    return eqn

from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, Reals,
                           Objective, Param, RangeSet, Var, exp, minimize, log)

from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'glpk')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

class exercise_prob(ConcreteModel):

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Ex P96')
        super(exercise_prob, self).__init__(*args, **kwargs)
        m = self

        """Set declarations"""
        I = m.I = RangeSet(0, 4, doc="continuous variables")
        J = m.J = RangeSet(1, 1, doc="discrete variables")

        # initial point information for discrete variables
        initY = {1: 1, 1: 1, 1: 1}
        # initial point information for continuous variables
        initX = {4: 0, 4: 0}

        """Variable declarations"""
        # DISCRETE VARIABLES
        Y = m.Y = Var(J, domain=Binary, initialize=initY)
        # CONTINUOUS VARIABLES
        X = m.X = Var(I, domain=Reals, initialize=initX)

        """Constraint definitions"""
        # CONSTRAINTS
        m.const1 = Constraint(expr= (X[1]-2)**2 - X[2] <= 0)
        m.const2 = Constraint(expr= X[1]-2*Y[1] >= 0)
        m.const3 = Constraint(expr= X[1]-X[2] -4*(1-Y[2]) <= 0)
        m.const4 = Constraint(expr= X[1]- (1-Y[1]) >= 0)
        m.const5 = Constraint(expr= X[2]-Y[2] >= 0)
        m.const6 = Constraint(expr= X[1]+X[2]-3*Y[3] >= 0)
        m.const7 = Constraint(expr= Y[1]+Y[2]+Y[3] >= 1)

        """Cost (objective) function definition"""
        m.cost = Objective(expr=Y[1] + 1.5*Y[2] + 0.5*Y[3] + X[1]**2 + X[2]**2,
         sense=minimize)
        
with SolverFactory('mindtpy') as opt:
            model = exercise_prob()
            print('\n Solving problem with Outer Approximation')
            opt.solve(model, strategy='OA', init_strategy='initial_binary',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0],
                      obj_bound=10)

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)

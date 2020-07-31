"""Solution of NLP subproblems."""
from __future__ import division

from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
                                                  add_int_cut)
from pyomo.contrib.mindtpy.util import add_feas_slacks
from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import (Constraint, Objective, TransformationFactory, Var,
                        minimize, value)
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning
from pyomo.contrib.mindtpy.objective_generation import generate_L2_objective_function

from pyomo.contrib.gdpopt.util import is_feasible

# Justin: Check this new argument always_solve_fix_nlp. Do we need it?
# Thought: this parameter does make things easier when solving fixed nlp in fp. 
# Maybe we can rename it to "fix_discrete" or something to make it more intuitive?
def solve_NLP_subproblem(solve_data, config, always_solve_fix_nlp=False):
    """
    Solves the fixed NLP (with fixed binaries)

    This function sets up the 'fixed_nlp' by fixing binaries, sets continuous variables to their intial var values,
    precomputes dual values, deactivates trivial constraints, and then solves NLP model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    always_solve_fix_nlp: boolean value
        Default to False; set to true when solving fixed NLP in feasibility pump.
    Returns
    -------
    fixed_nlp: Pyomo model
        fixed NLP from the model
    results: Pyomo results object
        result from solving the fixed NLP

    Note: Solves either fixed NLP (OA type methods) or feas-pump NLP

        Sets up local working model `fixed_nlp`
        Fixes binaries (OA) / Sets objective (feas-pump)
        Sets continuous variables to initial var values
        Precomputes dual values
        Deactivates trivial constraints
        Solves NLP model

        Returns the fixed-NLP model and the solver results
    """

    fixed_nlp = solve_data.working_model.clone()
    MindtPy = fixed_nlp.MindtPy_utils
    solve_data.nlp_iter += 1

    config.logger.info('NLP %s: Solve subproblem for fixed discretes.'
                        % (solve_data.nlp_iter,))
    TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)
    main_objective = next(fixed_nlp.component_data_objects(Objective, active=True))

    # Set up NLP
    if config.strategy == 'feas_pump':

        if main_objective.sense == 'minimize' and solve_data.UB != float('+inf'):
            fixed_nlp.increasing_objective_cut = Constraint(
                expr=fixed_nlp.MindtPy_utils.objective_value
                     <= solve_data.UB - config.feas_pump_delta*min(config.zero_tolerance, abs(solve_data.UB)))
        elif main_objective.sense == 'maximize' and solve_data.LB != float('-inf'):
            fixed_nlp.increasing_objective_cut = Constraint(
                expr=fixed_nlp.MindtPy_utils.objective_value
                     >= solve_data.LB + config.feas_pump_delta*max(config.zero_tolerance, abs(solve_data.LB)))
        
        main_objective.deactivate()
        MindtPy.feas_pump_nlp_obj = generate_L2_objective_function(
            fixed_nlp,
            solve_data.mip,
            discretes_only=True
        )

    MindtPy.MindtPy_linear_cuts.deactivate()
    fixed_nlp.tmp_duals = ComponentMap()
    # tmp_duals are the value of the dual variables stored before using deactivate trivial contraints
    # The values of the duals are computed as follows: (Complementary Slackness)
    #
    # | constraint | c_geq | status at x1 | tmp_dual (violation) |
    # |------------|-------|--------------|----------------------|
    # | g(x) <= b  | -1    | g(x1) <= b   | 0                    |
    # | g(x) <= b  | -1    | g(x1) > b    | g(x1) - b            |
    # | g(x) >= b  | +1    | g(x1) >= b   | 0                    |
    # | g(x) >= b  | +1    | g(x1) < b    | b - g(x1)            |
    evaluation_error = False
    for c in fixed_nlp.component_data_objects(ctype=Constraint, active=True,
                                              descend_into=True):
        # We prefer to include the upper bound as the right hand side since we are
        # considering c by default a (hopefully) convex function, which would make
        # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
        # if we don't have to
        rhs = c.upper if c.has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        # c_leq = 1 if c.has_ub else -1
        try:
            fixed_nlp.tmp_duals[c] = c_geq * max(
                0, c_geq*(rhs - value(c.body)))
        except (ValueError, OverflowError) as error:
            fixed_nlp.tmp_duals[c] = None
            evaluation_error = True
    if evaluation_error:
        for nlp_var, orig_val in zip(
                MindtPy.variable_list,
                solve_data.initial_var_values):
            if not nlp_var.fixed and not nlp_var.is_binary():
                nlp_var.value = orig_val
        # fixed_nlp.tmp_duals[c] = c_leq * max(
        #     0, c_leq*(value(c.body) - rhs))
        # TODO: change logic to c_leq based on benchmarking

    TransformationFactory('contrib.deactivate_trivial_constraints')\
        .apply_to(fixed_nlp, tmp=True, ignore_infeasible=True)
    # Solve the NLP
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            fixed_nlp, **config.nlp_solver_args, tee = True) # Justin don't forget to delete tee=True when done
    #Since the actual step of computing the nlp depends on the nlp_solver, it's possible that the inner loop of fp is missed here.
    return fixed_nlp, results

# The next few functions deal with handling the solution we get from the above NLP solver function
def handle_NLP_subproblem_optimal(fixed_nlp, solve_data, config):
    """
    This function copies the result of the NLP solver function ('solve_NLP_subproblem') to the working model, updates
    the bounds, adds OA and integer cuts, and then stores the new solution if it is the new best solution. This
    function handles the result of the latest iteration of solving the NLP subproblem given an optimal solution.

    Parameters
    ----------
    fixed_nlp: Pyomo model
        fixed NLP from the model
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    
    Note: Copies result to working model, updates bound, adds OA cut, no_good cut
    and increasing objective cut and stores best solution if new one is best and also calculate the duals
    """
    copy_var_list_values(
        fixed_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config,
        ignore_integrality=config.strategy == 'feas_pump')
    for c in fixed_nlp.tmp_duals:
        if fixed_nlp.dual.get(c, None) is None:
            fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
    dual_values = list(fixed_nlp.dual[c]
                       for c in fixed_nlp.MindtPy_utils.constraint_list)

    # main_objective = next(
    #     fixed_nlp.component_data_objects(Objective, active=True))  
    # this is different to original objective for feasibility pump

    # Justin: This might be neccesary to make the code work

    main_objective = next(
            solve_data.working_model.component_data_objects(
                Objective,
                active=True))  # this is different to original objective for feasibility pump



    # if OA-like or feas_pump converged, update Upper bound,
    # add no_good cuts and increasing objective cuts (feas_pump)
    if config.strategy in ['OA'] or (
        config.strategy == 'feas_pump'
        and feas_pump_converged(solve_data, config)
    ):
        if config.strategy == 'feas_pump':
            copy_var_list_values(solve_data.mip.MindtPy_utils.variable_list,
                                 solve_data.working_model.MindtPy_utils.variable_list,
                                 config)
            fix_nlp, fix_nlp_results = solve_NLP_subproblem(
                solve_data, config,
                always_solve_fix_nlp=True)
            assert fix_nlp_results.solver.termination_condition is tc.optimal, 'Feasibility pump fix-nlp subproblem not optimal'
            copy_var_list_values(fix_nlp.MindtPy_utils.variable_list,
                                 solve_data.working_model.MindtPy_utils.variable_list,
                                 config)
        if main_objective.sense == minimize:
            solve_data.UB = min(
                main_objective.expr(),
                solve_data.UB)
            solve_data.solution_improved = \
                solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)

            if solve_data.solution_improved and config.strategy == 'feas_pump':
                if solve_data.mip.find_component('increasing_objective_cut'):
                    solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                        increasing_objective_cut.set_value(
                            expr=solve_data.mip.MindtPy_utils.objective_value 
                                <= solve_data.UB - config.feas_pump_delta*min(config.zero_tolerance, abs(solve_data.UB)))
                else:
                   solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                        increasing_objective_cut = Constraint(expr=solve_data.mip.MindtPy_utils.objective_value
                            <= solve_data.UB - config.feas_pump_delta*min(config.zero_tolerance, abs(solve_data.UB)))
        else:
            solve_data.LB = max(
                main_objective.expr(),
                solve_data.LB)
            solve_data.solution_improved = \
                solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)

            if solve_data.solution_improved and config.strategy == 'feas_pump':
                if solve_data.mip.find_component('increasing_objective_cut'):
                    solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                        increasing_objective_cut.set_value(
                            expr=solve_data.mip.MindtPy_utils.objective_value
                                >= solve_data.LB + config.feas_pump_delta*max(config.zero_tolerance, abs(solve_data.LB)))
                else:
                   solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                        increasing_objective_cut = Constraint(expr=solve_data.mip.MindtPy_utils.objective_value 
                            >= solve_data.LB + config.feas_pump_delta*max(config.zero_tolerance, abs(solve_data.LB)))


    if main_objective.sense == minimize:
        solve_data.UB = min(value(main_objective.expr), solve_data.UB)
        solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
        solve_data.UB_progress.append(solve_data.UB)
    else:
        solve_data.solution_improved = False

    config.logger.info(
        'NLP {}: OBJ: {}  LB: {}  UB: {}'
        .format(solve_data.nlp_iter,
                main_objective.expr(),
                solve_data.LB, solve_data.UB))

    if solve_data.solution_improved:
        solve_data.best_solution_found = fixed_nlp.clone()
        # Justin: Check this assert. Document is_feasible
        # Thought: this shouldn't be a problem as this function(defined in gdpopt/util.py) is_feasible is based on tolerance in config.
        assert is_feasible(solve_data.best_solution_found, config), \
               "Best found model infeasible! There might be a problem with the precisions - the feaspump seems to have converged (error**2 <= integer_tolerance). " \
               "But the `is_feasible` check (error <= constraint_tolerance) doesn't work out"

    # Add the linear cut
    if config.strategy in ['OA', 'feas_pump']:
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config, ignore_integrality=config.strategy=='feas_pump')
        add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
    # elif config.strategy == 'PSC':
    #     add_psc_cut(solve_data, config)
    # elif config.strategy == 'GBD':
    #     add_gbd_cut(solve_data, config)
    
    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_no_good_cuts or config.strategy == 'feas_pump':
        add_int_cut(var_values, solve_data, config, feasible=True)

    config.call_after_subproblem_feasible(fixed_nlp, solve_data)

def handle_NLP_subproblem_infeasible(fixed_nlp, solve_data, config):
    """
    Solves feasibility problem and adds cut according to the specified strategy

    This function handles the result of the latest iteration of solving the NLP subproblem given an infeasible
    solution and copies the solution of the feasibility problem to the working model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # This adds an integer cut to the feasible_integer_cuts
    # ConstraintList, which is not activated by default. However, it
    # may be activated as needed in certain situations or for certain
    # values of option flags.
    # TODO try something else? Reinitialize with different initial
    # value?
    config.logger.info('NLP subproblem was locally infeasible.')
    for c in fixed_nlp.component_data_objects(ctype=Constraint):
        rhs = c.upper if c. has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        fixed_nlp.dual[c] = (c_geq
                             * max(0, c_geq * (rhs - value(c.body))))
    dual_values = list(fixed_nlp.dual[c]
                       for c in fixed_nlp.MindtPy_utils.constraint_list)

    # if config.strategy == 'PSC' or config.strategy == 'GBD':
    #     for var in fixed_nlp.component_data_objects(ctype=Var, descend_into=True):
    #         fixed_nlp.ipopt_zL_out[var] = 0
    #         fixed_nlp.ipopt_zU_out[var] = 0
    #         if var.has_ub() and abs(var.ub - value(var)) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zL_out[var] = 1
    #         elif var.has_lb() and abs(value(var) - var.lb) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zU_out[var] = -1

    if config.strategy == 'OA':
        config.logger.info('Solving feasibility problem')
        if config.initial_feas:
            # add_feas_slacks(fixed_nlp, solve_data)
            # config.initial_feas = False
            feas_NLP, feas_NLP_results = solve_NLP_feas(solve_data, config)
            copy_var_list_values(feas_NLP.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config)
            add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
    # Add an integer cut to exclude this discrete option
    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_no_good_cuts:
        # excludes current discrete option
        add_int_cut(var_values, solve_data, config)


def handle_NLP_subproblem_other_termination(fixed_nlp, termination_condition,
                                            solve_data, config):
    """
    Handles the result of the latest iteration of solving the NLP subproblem given a solution that is neither optimal
    nor infeasible.

    Parameters
    ----------
    termination_condition: Pyomo TerminationCondition
        the termination condition of the NLP subproblem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    if termination_condition is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        var_values = list(
            v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_integer_cuts:
            # excludes current discrete option
            add_int_cut(var_values, solve_data, config)
    else:
        raise ValueError(
            'MindtPy unable to handle NLP subproblem termination '
            'condition of {}'.format(termination_condition))


def solve_NLP_feas(solve_data, config):
    """
    Solves a feasibility NLP if the fixed_nlp problem is infeasible

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    feas_nlp: Pyomo model
        feasibility NLP from the model
    feas_soln: Pyomo results object
        result from solving the feasibility NLP
    """
    feas_nlp = solve_data.working_model.clone()
    add_feas_slacks(feas_nlp, config)

    MindtPy = feas_nlp.MindtPy_utils
    if MindtPy.find_component('objective_value') is not None:
        MindtPy.objective_value.value = 0

    next(feas_nlp.component_data_objects(Objective, active=True)).deactivate()
    for constr in feas_nlp.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        if constr.body.polynomial_degree() not in [0, 1]:
            constr.deactivate()

    MindtPy.MindtPy_feas.activate()
    if config.feasibility_norm == 'L1':
        MindtPy.MindtPy_feas_obj = Objective(
            expr=sum(s for s in MindtPy.MindtPy_feas.slack_var[...]),
            sense=minimize)
    elif config.feasibility_norm == 'L2':
        MindtPy.MindtPy_feas_obj = Objective(
            expr=sum(s*s for s in MindtPy.MindtPy_feas.slack_var[...]),
            sense=minimize)
    else:
        MindtPy.MindtPy_feas_obj = Objective(
            expr=MindtPy.MindtPy_feas.slack_var,
            sense=minimize)
    TransformationFactory('core.fix_integer_vars').apply_to(feas_nlp)
    with SuppressInfeasibleWarning():
        try:
            feas_soln = SolverFactory(config.nlp_solver).solve(
                feas_nlp, **config.nlp_solver_args)
        except (ValueError, OverflowError) as error:
            for nlp_var, orig_val in zip(
                    MindtPy.variable_list,
                    solve_data.initial_var_values):
                if not nlp_var.fixed and not nlp_var.is_binary():
                    nlp_var.value = orig_val
            feas_soln = SolverFactory(config.nlp_solver).solve(
                feas_nlp, **config.nlp_solver_args)
    subprob_terminate_cond = feas_soln.solver.termination_condition
    if subprob_terminate_cond is tc.optimal or subprob_terminate_cond is tc.locallyOptimal:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
    elif subprob_terminate_cond is tc.infeasible:
        raise ValueError('Feasibility NLP infeasible. '
                         'This should never happen.')
    else:
        raise ValueError(
            'MindtPy unable to handle feasibility NLP termination condition '
            'of {}'.format(subprob_terminate_cond))

    var_values = [v.value for v in MindtPy.variable_list]
    duals = [0 for _ in MindtPy.constraint_list]

    for i, c in enumerate(MindtPy.constraint_list):
        rhs = c.upper if c. has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        duals[i] = c_geq * max(
            0, c_geq * (rhs - value(c.body)))

    if value(MindtPy.MindtPy_feas_obj.expr) == 0:
        raise ValueError("Feasibility NLP problem is not feasible, check NLP solver output")

    return feas_nlp, feas_soln

#Thought: for this function, I 
def feas_pump_converged(solve_data, config):
    """
    Calculates the euclidean norm between the discretes in the mip and nlp models
    
    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns True if the solutin of the discrete variables in mip and nlp models converged.
    
    """
    distance = (sum((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if milp_var.is_binary()))

    return distance <= config.integer_tolerance

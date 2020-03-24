from pyomo.core.expr.logical_expr import LogicalXor

__all__ = ['BooleanVar', '_BooleanVarData', '_GeneralBooleanVarData', 'BooleanVarList', 'SimpleBooleanVar']


import logging
from weakref import ref as weakref_ref

from pyomo.common.timing import ConstructionTimer 
from pyomo.core.base.logicalvalue import LogicalValue, value, is_fixed
from pyomo.core.base.set_types import BooleanSet, IntegerSet, RealSet, Reals # needed?
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.sets import Set
from pyomo.core.base.util import is_functor


logger = logging.getLogger('pyomo.core')

class _BooleanVarData(ComponentData, LogicalValue):
    """
    This class defines the data for a single variable.

    Constructor Arguments:
        component   The BooleanVar object that owns this data.
    Public Class Attributes:
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        stale       A Boolean indicating whether the value of this variable is
                        legitimiate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.
    """
    __slots__ = () 
    def __init__(self, component=None):
        self._component = weakref_ref(component) if (component is not None) \
                          else None


    def is_binary(self):
        return True
        '''
        if self.domain.__class__ is BooleanSet:
            return True
        return isinstance(self.domain, BooleanSet)

		'''


    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def is_parameter_type(self):
        """Returns False because this is not a parameter object."""
        return False

    def is_variable_type(self):
        """Returns True because this is a variable."""
        return True

    def is_expression_type(self):
        """Returns False because this is not an expression"""
        return False

    def is_potentially_variable(self):
        """Returns True because this is a variable."""
        return True


    def set_value(self, val, valid=False):
        """
        Set the value of this numeric object, after
        validating its value. If the 'valid' flag is True,
        then the validation step is skipped.
        """
        if valid or self._valid_value(val):
            self.value = val
            self.stale = False
   
    def _valid_value(self, val, use_exception=True):
        """
        Validate the value.  If use_exception is True, then raise an
        exception.
        """
        ans = val is None or val in self.domain
        if not ans and use_exception:
            raise ValueError("Numeric value `%s` (%s) is not in "
                             "domain %s" % (val, type(val), self.domain))
        return ans

    def clear(self):
        self.value = None


    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    @property
    def value(self):
        """Return the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return the domain for this variable."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError
   
  

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        raise NotImplementedError

    def unfix(self):
        """Sets the fixed indicator to False."""
        raise NotImplementedError

    free=unfix

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Return the component name"""
        if self.fixed and compute_values:
            try:
                return str(self())
            except:
                pass
        if smap:
            return smap.getSymbol(self, labeler)
        return self.name
 
    def Xor(self, Y2):
        return LogicalXor(self, Y2)


class _GeneralBooleanVarData(_BooleanVarData):
    __slots__ = ('_value', '_domain', 'fixed', 'stale')
    #keep?
    def __init__(self, domain=Reals, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _BooleanVarData
        #   - ComponentData
        #   - LogicalValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._value = None
        self._domain = None
        self.fixed = False
        self.stale = True
        if hasattr(domain, 'bounds'):
            self._domain = domain
        elif domain is not None:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of one of %s, or an object that declares a method "
                "for bounds (like a Pyomo Set). Examples: NonNegativeReals, "
                "Integers, Binary" % (domain, (RealSet, IntegerSet, BooleanSet)))

    def __getstate__(self):
        state = super(_GeneralBooleanVarData, self).__getstate__()
        for i in _GeneralBooleanVarData.__slots__:
            state[i] = getattr(self, i)
        return state


    @property
    def value(self):
        """Return the value for this variable."""
        return self._value
    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        self._value = val

    @property
    def domain(self):
        """Return the domain for this variable."""
        return self._domain
    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        if hasattr(domain, 'bounds'):
            self._domain = domain
        else:
            raise ValueError(
                "%s is not a valid domain. Variable domains must be an "
                "instance of one of %s, or an object that declares a method "
                "for bounds (like a Pyomo Set). Examples: NonNegativeReals, "
                "Integers, Binary" % (domain, (RealSet, IntegerSet, BooleanSet)))

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        self.fixed = True
        if len(val) == 1:
            self.value = val[0]
        elif len(val) > 1:
            raise TypeError("fix expected at most 1 arguments, got %d" % (len(val)))

    def unfix(self):
        """Sets the fixed indicator to False."""
        self.fixed = False

    free = unfix



@ModelComponentFactory.register("Decision variables.")
class BooleanVar(IndexedComponent):
    _ComponentDataClass = _GeneralBooleanVarData
    
    def __new__(cls, *args, **kwds):
        if cls != BooleanVar:
            return super(BooleanVar, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return SimpleBooleanVar.__new__(SimpleBooleanVar)
        else:
            return IndexedBooleanVar.__new__(IndexedBooleanVar) 

    def __init__(self, *args, **kwd):
        initialize = kwd.pop('initialize', None)
        initialize = kwd.pop('rule', initialize)
        domain = kwd.pop('within', Reals)
        domain = kwd.pop('domain', domain)
        self._dense = kwd.pop('dense', True)


        kwd.setdefault('ctype', BooleanVar)
        IndexedComponent.__init__(self, *args, **kwd)
        self._domain_init_value = None
        self._domain_init_rule = None
        if is_functor(domain):
            self._domain_init_rule = domain
        else:
            self._domain_init_value = domain
            
        self._value_init_value = None
        self._value_init_rule = None

        if  is_functor(initialize) and (not isinstance(initialize,LogicalValue)):
            self._value_init_rule = initialize
        else:
            self._value_init_value = initialize

    def is_expression_type(self):
        """Returns False because this is not an expression"""
        return False

    def flag_as_stale(self):
        """
        Set the 'stale' attribute of every variable data object to True.
        """
        for boolvar_data in itervalues(self._data):
            boolvar_data.stale = True

    def get_values(self, include_fixed_values=True):
        """
        Return a dictionary of index-value pairs.
        """
        if include_fixed_values:
            return dict((idx, vardata.value)
                            for idx, vardata in iteritems(self._data))
        return dict((idx, vardata.value)
                            for idx, vardata in iteritems(self._data)
                                                if not vardata.fixed)

    extract_values = get_values

    def set_values(self, new_values, valid=False):
        """
        copied from var.py
        Set the values of a dictionary.

        The default behavior is to validate the values in the
        dictionary.
        """
        for index, new_value in iteritems(new_values):
            self[index].set_value(new_value, valid)


    def construct(self, data=None):
        """Construct this component."""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):   #pragma:nocover
            try:
                name = str(self.name)
            except:
                name = type(self)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Constructing Variable, name=%s, from data=%s"
                    % (name, str(data)))

        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True

        #
        # Construct _BooleanVarData objects for all index values
        #
        if not self.is_indexed():
            self._data[None] = self
            self._initialize_members((None,))
        elif self._dense:
            # This loop is optimized for speed with pypy.
            # Calling dict.update((...) for ...) is roughly
            # 30% slower
            self_weakref = weakref_ref(self)
            for ndx in self._index:
                cdata = self._ComponentDataClass(
                    domain=self._domain_init_value, component=None)
                cdata._component = self_weakref
                self._data[ndx] = cdata
                #self._initialize_members((ndx,))
            self._initialize_members(self._index)
        timer.report()

    def add(self, index):
        """Add a variable with a particular index."""
        return self[index]

    #
    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    def _getitem_when_not_present(self, index):
        """Returns the default component data value."""
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(
                self._domain_init_value, component=self)
        self._initialize_members((index,))
        return obj
    def _setitem_when_not_present(self, index, value):
        """Perform the fundamental component item creation and storage.

        BooleanVar overrides the default implementation from IndexedComponent
        to enforce the call to _initialize_members.

        """
        obj = self._getitem_when_not_present(index)
        try:
            return obj.set_value(value)
        except:
            del self._data[index]
            raise
    #keep?
    def _initialize_members(self, init_set):
        """Initialize variable data for all indices in a set."""
        # TODO: determine if there is any advantage to supporting init_set.
        # Preliminary tests indicate that there isn't a significant speed
        # difference to using the set form (used in dense vector
        # construction).  Getting rid of it could simplify _setitem and
        # this method.
        # #0-0 Do I need to care about the above?
        # Initialize domains
        #
        if self._domain_init_rule is not None:
            #
            # Initialize domains with a rule
            #
            if self.is_indexed():
                for ndx in init_set:
                    self._data[ndx].domain = \
                        apply_indexed_rule(self,
                                           self._domain_init_rule,
                                           self._parent(),
                                           ndx)
            else:
                self.domain = self._domain_init_rule(self._parent())
        else:
            if self.is_indexed():
                # Optimization: It is assumed self._domain_init_value
                #               is used when the _GeneralBooleanVarData objects
                #               are created. This avoids an unnecessary
                #               loop over init_set, which can significantly
                #               speed up construction of variables with large
                #               index sets.
                pass
            else:
                # the above optimization does not apply for
                # singleton objects (trying to do so breaks
                # some of the pickle tests)
                self.domain = self._domain_init_value

        #
        # Initialize values
        #
        if self._value_init_rule is not None:
            #
            # Initialize values with a rule
            #
            if self.is_indexed():
                for key in init_set:
                    vardata = self._data[key]
                    val = apply_indexed_rule(self,
                                             self._value_init_rule,
                                             self._parent(),
                                             key)
                    val = value(val)
                    vardata.set_value(val)
            else:
                val = self._value_init_rule(self._parent())
                val = value(val)
                self.set_value(val)
        elif self._value_init_value is not None:
            #
            # Initialize values with a value
            if self._value_init_value.__class__ is dict:
                for key in init_set:
                    # Skip indices that are not in the
                    # dictionary. This arises when
                    # initializing VarList objects with a
                    # dictionary.
                    # What does this continue do here?
                    if not key in self._value_init_value:
                        continue
                    val = self._value_init_value[key]
                    vardata = self._data[key]
                    vardata.set_value(val)
            else:
                val = value(self._value_init_value)
                for key in init_set:
                    vardata = self._data[key]
                    vardata.set_value(val)

    #keep?
    def _pprint(self):
        """
            Print component information.
        """
        pass
        '''
        return ( [("Size", len(self)),
                  ("Index", self._index if self.is_indexed() else None),
                  ],
                 iteritems(self._data),
                 ( "Lower","Value","Upper","Fixed","Stale","Domain"),
                 lambda k, v: [ value(v.lb),
                                v.value,
                                value(v.ub),
                                v.fixed,
                                v.stale,
                                v.domain
                                ]
                 )
                 '''

#keep?
class SimpleBooleanVar(_GeneralBooleanVarData, BooleanVar):
    
    """A single variable."""
    def __init__(self, *args, **kwd):
        _GeneralBooleanVarData.__init__(self,
                                 domain=None,
                                 component=self)
        BooleanVar.__init__(self, *args, **kwd)

    """
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our
    # get/set state methods rely on super() to traverse the MRO, this
    # will automatically pick up both the Component and Data base classes.
    #

    #
    # Override abstract interface methods to first check for
    # construction
    #

    # NOTE: that we can't provide these errors for
    # fixed and stale because they are attributes
    """

    @property
    def value(self):
        """Return the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fget(self)
        raise ValueError(
            "Accessing the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.value.fset(self, val)
        raise ValueError(
            "Setting the value of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))


    @property
    def domain(self):
        return _GeneralBooleanVarData.domain.fget(self)

    @domain.setter
    def domain(self, domain):
        """Set the domain for this variable."""
        if self._constructed:
            return _GeneralBooleanVarData.domain.fset(self, domain)
        raise ValueError(
            "Setting the domain of variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set."
            % (self.name))


    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        if self._constructed:
            return _GeneralBooleanVarData.fix(self, *val)
        raise ValueError(
            "Fixing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    def unfix(self):
        """Sets the fixed indicator to False."""
        if self._constructed:
            return _GeneralBooleanVarData.unfix(self)
        raise ValueError(
            "Freeing variable '%s' "
            "before the Var has been constructed (there "
            "is currently nothing to set)."
            % (self.name))

    free=unfix

class IndexedBooleanVar(BooleanVar):
    """An array of variables."""

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        for boolean_vardata in itervalues(self):
            boolean_vardata.fix(*val)

    def unfix(self):
        """Sets the fixed indicator to False."""
        for boolean_vardata in itervalues(self):
            boolean_vardata.unfix()

    @property
    def domain(self):
        raise AttributeError(
            "The domain is not an attribute for IndexedBooleanVar. It "
            "can be set for all indices using this property setter, "
            "but must be accessed for individual variables in this container.")
    @domain.setter
    def domain(self, domain):
        """Sets the domain for all variables in this container."""
        for boolean_vardata in itervalues(self):
            boolean_vardata.domain = domain

    free=unfix
    

@ModelComponentFactory.register("List of decision variables.")
class BooleanVarList(IndexedBooleanVar):
    """
    Variable-length indexed variable objects used to construct Pyomo models.
    """

    def __init__(self, **kwds):
        args = (Set(),)
        IndexedBooleanVar.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """Construct this component."""
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing variable list %s", self.name)

        # We need to ensure that the indices needed for initialization are
        # added to the underlying implicit set.  We *could* verify that the
        # indices in the initialization dict are all sequential integers,
        # OR we can just add the correct number of sequential integers and
        # then let _validate_index complain when we set the value.
        if self._value_init_value.__class__ is dict:
            for i in xrange(len(self._value_init_value)):
                self._index.add(i+1)
        super(BooleanVarList,self).construct(data)
        # Note that the current Var initializer silently ignores
        # initialization data that is not in the underlying index set.  To
        # ensure that at least here all initialization data is added to the
        # VarList (so we get potential domain errors), we will re-set
        # everything.
        if self._value_init_value.__class__ is dict:
            for k,v in iteritems(self._value_init_value):
                self[k] = v

    def add(self):
        """Add a variable to this list."""
        next_idx = len(self._index) + 1
        self._index.add(next_idx)
        return self[next_idx]





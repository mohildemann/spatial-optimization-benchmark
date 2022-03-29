class Problem:
    _id = 0
    def __init__(self,
                 name,
                 nr_objectives,
                 benchmarks,
                 objective_names,
                 objective_descriptions,
                 objective_functions,
                 constraint_descriptions,
                 validation_functions,
                 mathematical_formulation,
                 plot_layout,
                 plot_function_solution,
                 plot_background_trace,
                 plot_geographical_center,
                 plot_function_additional_trace):
        self.problem_id = Problem._id
        Problem._id += 1
        self.representation = name
        self.nr_objectives = nr_objectives
        self.benchmarks = benchmarks
        self.objective_names = objective_names
        self.objective_descriptions = objective_descriptions
        self.objective_functions = objective_functions
        self.constraint_descriptions = constraint_descriptions
        self.validation_functions = validation_functions
        self.mathematical_formulation = mathematical_formulation
        self.plot_layout = plot_layout
        self.plot_function_solution = plot_function_solution
        self.plot_background_trace = plot_background_trace
        self.geographical_center = plot_geographical_center
        self.plot_function_additional_trace = plot_function_additional_trace

def problem_dictionary(self):
    return
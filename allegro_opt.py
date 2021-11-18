from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from clearml import Task 

def main():
    task = Task.init(project_name='Hyper-Parameter Search', 
                    task_name='Hyper-Parameter Optimization')

    optimizer = HyperParameterOptimizer(
        base_task_id="7474bfaca1e64cd1b432e33a4783d51e",  
        # setting the hyper-parameters to optimize
            hyper_parameters=[
                UniformIntegerParameterRange('batch_size', min_value=8, max_value=32, step_size=8),
                UniformParameterRange('dropout', min_value=0.1, max_value=0.5, step_size=0.1),
                UniformParameterRange('learning_rate', min_value=1e-5, max_value=1e-3, step_size=1e-5),
            ],
            
        # setting the objective metric we want to maximize/minimize
            objective_metric_title='Loss',
            objective_metric_series='test:',
            objective_metric_sign='min',  
            
        # setting optimizer 
        optimizer_class=OptimizerOptuna,
        
        # Configuring optimization parameters
            max_number_of_concurrent_tasks=5,  
            optimization_time_limit=60*3, 
            compute_time_limit=60*3*10, 
            total_max_jobs=20,  
            min_iteration_per_job=15000,  
            max_iteration_per_job=150000)


    optimizer.set_report_period(1) # setting the time gap between two consecutive reports
    optimizer.start()
    optimizer.wait() # wait until process is done
    optimizer.stop() # make sure background optimization stopped

if __name__ == "__main__":
    main()

from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from clearml import Task 

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

def main():
    
    task = Task.init(project_name='Hyper-Parameter Search', 
                    task_name='Hyper-Parameter Optimization',                    
                    task_type=Task.TaskTypes.optimizer,
                    reuse_last_task_id=False)
    
    # experiment template to optimize in the hyper-parameter optimization
    args = {
        'template_task_id': None,
        'run_as_service': False,
    }
    args = task.connect(args)
    
    # Get the template task experiment that we want to optimize
    if not args['template_task_id']:
        args['template_task_id'] = Task.get_task(
            project_name='protein_binding', task_name='deep-TCELL').id

    optimizer = HyperParameterOptimizer(
        base_task_id="b0c204b3865d4ff0a0fed6cc02a3c0b1",  
        
        # setting the hyper-parameters to optimize
            hyper_parameters=[
                UniformIntegerParameterRange('batch_size', min_value=8, max_value=32, step_size=8),
                UniformParameterRange('dropout', min_value=0.1, max_value=0.5, step_size=0.1),
                UniformParameterRange('learning_rate', min_value=1e-5, max_value=1e-3, step_size=1e-5),
            ],
            
        # setting the objective metric we want to maximize/minimize
            objective_metric_title='Loss',
            objective_metric_series='test',
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


    optimizer.set_report_period(0.2) # setting the time gap between two consecutive reports
    optimizer.start(job_complete_callback=job_complete_callback)
    optimizer.wait() # wait until process is done
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()
    
    print('We are done, good bye')

if __name__ == "__main__":
    main()

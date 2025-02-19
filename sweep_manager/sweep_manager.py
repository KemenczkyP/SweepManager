import json

import mlflow
import optuna


class SweepManager:
    def __init__(self, sweep_id, num_trials=100,
                 optuna_direction: str="minimize",  optuna_n_jobs: int=1,):
        """
        The constructor of the SweepManager class
        Args:
            sweep_id (str): The sweep id
            num_trials (int): The number of trials
            optuna_direction (str): The direction of the optimization. It can be either "minimize" or "maximize"
            optuna_n_jobs (int): The number of jobs for the optimization
        Raises:
            AssertionError: If the sweep id is not a string
            AssertionError: If the number of trials is not an integer
        """

        assert isinstance(sweep_id, str), "The sweep id should be a string"
        assert isinstance(num_trials, int), "The number of trials should be an integer"
        assert optuna_direction in ["minimize", "maximize"], "The direction of the optimization is not valid"
        assert isinstance(optuna_n_jobs, int), "The number of jobs for the optimization should be an integer"

        self.sweep_id = sweep_id
        self.__mlflow_configured = False
        self.num_trials = num_trials
        self.optuna_direction = optuna_direction
        self.optuna_n_jobs = optuna_n_jobs

    def configure_MLFlow(self, experiment_name: str, tracking_uri: str):
        """
        Configure MLFlow. This function should be called before running the objective function.
        Args:
            experiment_name (str): The name of the experiment
            tracking_uri (str): The tracking URI of the MLFlow server.
        Returns:
            None
        """

        assert isinstance(experiment_name, str), "The experiment name should be a string"

        # set server URI
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.__mlflow_configured = True

    def run_optuna_with_mlflow(self, objective_function):
        # Start an MLFlow run
        # Create a study object and optimize the objective function
        study = optuna.create_study(direction=self.optuna_direction)
        study.optimize(objective_function, n_trials=self.num_trials, n_jobs=self.optuna_n_jobs, gc_after_trial=True)

        # Log the best parameters and the best value
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_value", study.best_value)

        # Optionally, log the entire study as an artifact
        # This requires the study to be saved to a file first
        # study.trials_dataframe().to_csv("study_results.csv")
        # mlflow.log_artifact("study_results.csv")

    def _decode_hyperparameters(self, trial, hyperparameters: dict):

        optuna_hyperparameters = {}

        # Decode the hyperparameters from the sweep manager
        for hyperparameter_name, hyperparameter_settings in hyperparameters.items():
            # Check if the necessary fields are defined
            assert "type" in hyperparameter_settings, "The type of the hyperparameter is not defined"
            assert "default" in hyperparameter_settings, "The default value is not defined"

            # Get the hyperparameter settings
            hyperparameter_type: str = hyperparameter_settings.get("type")
            hyperparameter_default: list = hyperparameter_settings.get("default")
            hyperparameter_iter: list = hyperparameter_settings.get("iter")
            hyperparameter_range: list = hyperparameter_settings.get("range")

            # Check if the hyperparameter is categorical. If it is, the iter should be defined!
            if hyperparameter_type == "categorical":
                if hyperparameter_iter is None and hyperparameter_range is not None:
                    raise ValueError("The hyperparameter type is categorical, but the iter or range is not defined")

            if hyperparameter_iter:
                optuna_hyperparameters[hyperparameter_name] = trial.suggest_categorical(hyperparameter_name,
                                                                                        hyperparameter_iter)
            elif hyperparameter_range:
                assert len(hyperparameter_range) == 2, "The range should be a list of two elements"
                if hyperparameter_type == "int":
                    optuna_hyperparameters[hyperparameter_name] = trial.suggest_int(hyperparameter_name,
                                                                                    hyperparameter_range[0],
                                                                                    hyperparameter_range[1])
                elif hyperparameter_type == "float":
                    optuna_hyperparameters[hyperparameter_name] = trial.suggest_float(hyperparameter_name,
                                                                                      hyperparameter_range[0],
                                                                                      hyperparameter_range[1])
                else:
                    raise ValueError("The hyperparameter type is not valid")
            else:
                optuna_hyperparameters[hyperparameter_name] = hyperparameter_default

        return optuna_hyperparameters, trial

    def get_objective(self, hyperparameters_path: str,
                      train_function):
        def objective(trial):
            with mlflow.start_run() as run:
                assert self.__mlflow_configured, "MLFlow is not configured. Please configure it before running the objective function"

                # Define your hyperparameters
                hyperparameters_raw = json.load(open(hyperparameters_path, "r"))
                hyperparameters_raw = hyperparameters_raw.get(
                    "hyperparameters")  # There are `help` and `hyperparameters` fields
                hyp_set_params, trial = self._decode_hyperparameters(trial, hyperparameters_raw)

                # # log the results to MLFlow
                for param_name, param_value in hyp_set_params.items():
                    mlflow.log_param(param_name, param_value)
                results = train_function(hyp_set_params, run.info.run_id)
            mlflow.end_run()

            return results

        return objective

"""
HPO Quickstart with PyTorch
===========================
This tutorial optimizes the model in `official PyTorch quickstart`_ with auto-tuning.

The tutorial consists of 4 steps:

1. Modify the model for auto-tuning.
2. Define hyperparameters' search space.
3. Configure the experiment.
4. Run the experiment.

.. _official PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""
from nni.experiment import Experiment

# %%
# Step 1: Prepare the model
# -------------------------
# In first step, we need to prepare the model to be tuned.
#
# The model should be put in a separate script.
# It will be evaluated many times concurrently,
# and possibly will be trained on distributed platforms.
#
# In this tutorial, the model is defined in :doc:`model.py <model>`.
#
# In short, it is a PyTorch model with 3 additional API calls:
#
# 1. Use :func:`nni.get_next_parameter` to fetch the hyperparameters to be evalutated.
# 2. Use :func:`nni.report_intermediate_result` to report per-epoch accuracy metrics.
# 3. Use :func:`nni.report_final_result` to report final accuracy.
#
# Please understand the model code before continue to next step.

# %%
# Step 2: Define search space
# ---------------------------
# In model code, we have prepared 3 hyperparameters to be tuned:
# *features*, *lr*, and *momentum*.
#
# Here we need to define their *search space* so the tuning algorithm can sample them in desired range.
#
# Assuming we have following prior knowledge for these hyperparameters:
#
# 1. *features* should be one of 128, 256, 512, 1024.
# 2. *lr* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
# 3. *momentum* should be a float between 0 and 1.
#
# In NNI, the space of *features* is called ``choice``;
# the space of *lr* is called ``loguniform``;
# and the space of *momentum* is called ``uniform``.
# You may have noticed, these names are derived from ``numpy.random``.
#
# For full specification of search space, check :doc:`the reference </hpo/search_space>`.
#
# Now we can define the search space as follow:

search_space = {
    # 'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
    # 'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

# %%
# Step 3: Configure the experiment
# --------------------------------
# NNI uses an *experiment* to manage the HPO process.
# The *experiment config* defines how to train the models and how to explore the search space.
#
# In this tutorial we use a *local* mode experiment,
# which means models will be trained on local machine, without using any special training platform.

experiment = Experiment("local")

# %%
# Now we start to configure the experiment.
#
# Configure trial code
# ^^^^^^^^^^^^^^^^^^^^
# In NNI evaluation of each hyperparameter set is called a *trial*.
# So the model script is called *trial code*.
experiment.config.trial_command = "python main.py --enable_nni"
experiment.config.trial_code_directory = "."
# %%
# When ``trial_code_directory`` is a relative path, it relates to current working directory.
# To run ``main.py`` in a different path, you can set trial code directory to ``Path(__file__).parent``.
# (`__file__ <https://docs.python.org/3.10/reference/datamodel.html#index-43>`__
# is only available in standard Python, not in Jupyter Notebook.)
#
# .. attention::
#
#     If you are using Linux system without Conda,
#     you may need to change ``"python model.py"`` to ``"python3 model.py"``.

# %%
# Configure search space
# ^^^^^^^^^^^^^^^^^^^^^^
experiment.config.search_space = search_space

# %%
# Configure tuning algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we use :doc:`TPE tuner </hpo/tuners>`.
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "maximize"

# %%
# Configure how many trials to run
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 2 sets at a time.
experiment.config.max_trial_number = 4
experiment.config.trial_concurrency = 2
experiment.config.max_experiment_duration = "1h"
# %%
# You may also set ``max_experiment_duration = '1h'`` to limit running time.
#
# If neither ``max_trial_number`` nor ``max_experiment_duration`` are set,
# the experiment will run forever until you press Ctrl-C.
#
# .. note::
#
#     ``max_trial_number`` is set to 10 here for a fast example.
#     In real world it should be set to a larger number.
#     With default config TPE tuner requires 20 trials to warm up.

# %%
# Step 4: Run the experiment
# --------------------------
# Now the experiment is ready. Choose a port and launch it. (Here we use port 8080.)
#
# You can use the web portal to view experiment status: http://localhost:8080.
experiment.run(8084)

# %%
# After the experiment is done
# ----------------------------
# Everything is done and it is safe to exit now. The following are optional.
#
# If you are using standard Python instead of Jupyter Notebook,
# you can add ``input()`` or ``signal.pause()`` to prevent Python from exiting,
# allowing you to view the web portal after the experiment is done.

# input('Press enter to quit')
experiment.save()

experiment.stop()

# %%
# :meth:`nni.experiment.Experiment.stop` is automatically invoked when Python exits,
# so it can be omitted in your code.
#
# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` to restart web portal.
#
# .. tip::
#
#     This example uses :doc:`Python API </reference/experiment>` to create experiment.
#
#     You can also create and manage experiments with :doc:`command line tool <../hpo_nnictl/nnictl>`.


# def test_seed(script_name: str, dataset='SU', enable_nni=False, num_trials=1000,
#               node_features='degree_bin'):
#     seeds = []
#     num_trials = 10 if enable_nni else num_trials
#     print(f'running {num_trials} trials')
#     for i in range(num_trials):
#         seeds.append(random.randint(100000, 10000000))

#     default_param = {
#         'dataset_name': dataset,
#         'node_features': node_features,
#         'weight_decay': 5e-4,
#         'epochs': 100,
#         'n_MLP_layer': 1,
#         'n_GNN_layers': 3,
#         # 'hidden_dim': 360,
#         # 'edge_emb_dim':256,
#         'lr': 0.001,
#     }

#     sp = SP.SimpleParam(default=default_param)
#     params = sp(from_='./src/nni_configs/search_space.json', preprocess_nni=False)

#     param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])

#     cmd = f'python {script_name} {param_str}'
#     cmd += ' --enable_nni' if enable_nni else ''
#     print(cmd)
#     os.system(cmd)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--target', type=str, default='main.py')
#     parser.add_argument('--enable_nni', action='store_true')
#     parser.add_argument('--dataset', type=str, default='SU')
#     parser.add_argument('--trials', type=int, default=1000)
#     parser.add_argument('--node_features', type=str,
#                         choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj'],
#                         default='adj')
#     args = parser.parse_args()

#     cwd = os.getcwd()
#     print(cwd)

#     test_seed(args.target, dataset=args.dataset, enable_nni=args.enable_nni, num_trials=args.trials,
#               node_features=args.node_features)

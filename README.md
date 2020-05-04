# adaptive-spreading

Code for the paper "Learning Controllers for Adaptive Spreading of Carbon Fiber Tows".
Download the data (preprocessed) and a pretrained process model here:
https://figshare.com/s/1a3e9b1ac16362b46cf9

## Run experiments

* Download the data and adjust the data path if necessary (and other in-/output paths) in the `utils/paths.py` file.  
The data is used for both, the Process Model and the Process Control Model.

### Process Model

* To train a feedforward neural network or a random forest, run 
`python -m process_model.start_ff` or `python -m process_model.start_rf`

* Results can be found in the generated log file (per default in the `logs` directory)

* When training a neural network, in addition, a tensorboard file is generated (in the `runs` directory). 

### Process Control Model

* To start the neuroevolution, run `python -m process_control_model.start_neuroevolution`

    * Parameters can be set in the source code or passed as json-file
(see `process_control_model/params/example_ne_param_file.json` for details)
    * Please adjust the type of the process_model used as backend (i.e. "nn" or "rf") and
    specify the path to the pre-trained model.
    * Results are logged to a file in the directory `process_control_model/logs`

* The baseline can be run as follows
`python -m process_control_model.fixed_setup`.
Log files can be found in the same directory as the neuroevolution logs.

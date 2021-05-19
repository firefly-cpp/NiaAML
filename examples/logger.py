from niaaml.logger import Logger

"""
This example presents how to use Logger class individually.
"""

# instatiate instance with verbose mode
logger = Logger(verbose=True)

# in verbose mode, all of the call functions should show their output
logger.log_progress("progress")
logger.log_pipeline("pipeline")
logger.log_optimization_error("optimization error")

print("-------------------------")

# in this case only log_progress function's call is going to show the output
logger = Logger()
logger.log_progress("progress")
logger.log_pipeline("pipeline")
logger.log_optimization_error("optimization error")

print("-------------------------")

# you may also output logs to some log file
logger = Logger(verbose=True, output_file="log_output")
logger.log_progress("progress")
logger.log_pipeline("pipeline")
logger.log_optimization_error("optimization error")

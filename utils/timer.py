# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:31:23 2024

@author: Markos Athanasiadis
"""

import numpy as np
import time

def timer(start_timestamp):
    """
    Timer Function to Calculate and Print Elapsed Time in a Human-Readable Format.

    This function calculates the elapsed time since the given `start_timestamp` and prints
    it in hours, minutes, and seconds format. It is useful for tracking the execution time 
    of code blocks or programs.

    Parameters:
    -----------
    start_timestamp : float
        The time point (in seconds since the epoch) at which the process or task began. 
        This is usually the output of `time.time()` when the task is initialized.

    Output:
    -------
    None : This function prints the elapsed time directly to the console.

    """
    
    stop_timestamp = time.time()  # Capture the current time
    elapsed_seconds = stop_timestamp - start_timestamp  # Time difference in seconds
    
    # Convert the time difference to an integer and round it
    elapsed_seconds = int(np.round(elapsed_seconds))
    
    if elapsed_seconds < 60:
        # If elapsed time is less than 1 minute, print seconds
        print(f'--- {elapsed_seconds} seconds ---')
    else:
        # If elapsed time is at least 1 minute, calculate minutes and seconds
        elapsed_minutes = elapsed_seconds // 60
        remaining_seconds = elapsed_seconds % 60
        
        if elapsed_minutes < 60:
            # If elapsed time is less than 1 hour, print minutes and seconds
            print(f'--- {elapsed_minutes} minutes and {remaining_seconds} seconds ---')
        else:
            # If elapsed time is at least 1 hour, calculate hours, minutes, and seconds
            elapsed_hours = elapsed_minutes // 60
            remaining_minutes = elapsed_minutes % 60
            print(f'--- {elapsed_hours} hours and {remaining_minutes} minutes and {remaining_seconds} seconds ---')





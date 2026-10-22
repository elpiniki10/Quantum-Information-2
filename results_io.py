"""Saves results row by row in a txt file, with an autosave every x rows.
If the program crashes or is stopped, running it again continues from the last autosave."""

import os
import numpy as np

def rows_done(path:str):
    """Number of complete result rows already in the file (0 if it doesn't exist)."""
    if not os.path.exists(path):#  if the file doesn't exist, 0 rows are done
        return 0
    with open(path,"r",encoding="utf-8") as f:
        text=f.read()
    if text and not text.endswith("\n"):#if text not empy->true,if text doesnt end with newline->true(it didnt crash)
        text=text[:text.rfind("\n")+1]# Remove the incomplete last line
        with open(path,"w",encoding="utf-8") as f:# Reopen file for writing
            f.write(text)# Save only the complete lines
    return sum(1 for line in text.splitlines() if line and not line.startswith("#"))# Count non-empty, non-header line

def append_rows(path:str,rows:list,header:list[str]):
    """Appends rows to the file (header written once) and forces them onto the disk"""
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)# create the folder if it is missing.
    new=not os.path.exists(path) or os.path.getsize(path)==0#Decide whether this is a new file.True if file does not exist.True if file exists but is empty
    with open(path,"a",encoding="utf-8") as f:# Open file in append mode
        if new:# Only write header for a new file
            f.write("# "+"\t".join(header)+"\n")# Write column names separated by tabs
        for row in rows:# Go through each result row
            f.write("\t".join(f"{x:.10g}" for x in row)+"\n")# Write the row to the file.Convert numbers to text with 10 digits.Move to the next line
        f.flush()# Push buffered data out of Python
        os.fsync(f.fileno())# Force the data to be written to disk

def run_checkpointed(path:str,header:list[str],n_total:int,compute_row,every:int=1000):
    """Calls compute_row(i) for i = 0..n_total-1 and saves the results in path.
    compute_row returns a list of numbers (one per header column).
    Rows already in the file are skipped, so a crashed run resumes where it stopped."""
    if os.path.exists(path) and os.path.getsize(path)>0:# An old file must have the same columns
        with open(path,"r",encoding="utf-8") as f:
            if f.readline().rstrip("\n")!="# "+"\t".join(header):
                raise ValueError(f"{path} has different columns, delete it to start a fresh run")
    start=rows_done(path)# Find how many rows have already been saved
    if start>=n_total:# Check whether all rows are already finished
        print(f"{path}: all {n_total} results already saved")
        return
    if start>0:# Check whether we are continuing an old calculation
        print(f"{path}: resuming from result {start}/{n_total}")
    buffer=[]# Temporary list for storing new rows
    for i in range(start,n_total):# Loop over all remaining row numbers
        buffer.append(compute_row(i))# Calculate row i and store it temporarily
        if len(buffer)>=every:# Check whether it's time to save
            append_rows(path,buffer,header)# Save the buffered rows to the file
            buffer=[]# Empty the buffer after saving
            print(f"{path}: autosaved {i+1}/{n_total}")# Show progress to the user
    if buffer:# Check if some rows remain unsaved
        append_rows(path,buffer,header)# Save the remaining rows
    print(f"{path}: done, {n_total} results saved")# Tell the user the calculation is complete

def load_results(path:str):
    """Loads the txt file; columns are accessed by name, e.g. data['p']"""
    return np.genfromtxt(path,names=True,delimiter="\t")

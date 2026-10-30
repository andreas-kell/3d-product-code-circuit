# 3d Product Code Circuit

This project provides a python script, which reproduces an interleaved syndrome extraction circuit
for the 3D product code.

## Installation
Install the `uv` package manager and install the dependencies with `uv sync`.

## Usage
Run the python script with `uv run ./generate_stim_circuit.py`.

## Expected results
The script will create the `results` folder and in it
- the stim circuit file `circuit.txt`,
- the stim detector error model file `detector_error_model.txt`,
- and a visualization of the circuit `circuit.svg`.

If the circuit is too large to be displayed properly in a browser, a workaround is to add `width`
and `height` attributes to the first svg tag. Example:
```svg
<svg viewBox="0 0 705120 57440" width="70000" height="6000" version="1.1" xmlns="http://www.w3.org/2000/svg">
```
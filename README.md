## MCMC sampler presented in the paper "Layering-MCMC for Structure Learning in Bayesian Networks" (UAI 2020)

### Contents

`layeringMCMC.py`: All the functionality required for layering-MCMC.

`boston.jkl`: An example score file computed from the [Boston housing dataset](http://lib.stat.cmu.edu/datasets/boston) with BGe scores.

`COPYING`: Copyright and license information. 

`README.md`: This file.

### Dependencies and installation

Requires Python 3 (tested with 3.7.6) and Numpy (1.18.1).

No installation necessary, just clone the repository and run

`$ python layeringMCMC.py boston.jkl 8 3 1000 0`

### Arguments

All arguments are positional, in the following order
1. Path to local scores (jkl format),
2. M parameter,
3. max-indegree, 
4. number of steps to take in MCMC,
5. random seed.

### Output

The script prints out a line for each MCMC step with the following comma separated content

1. **M-Layering** 
    
    Each node label separated by white space and each layer separated by `|`.
    
2. **M-Layering score**
    
    Sum of scores of DAGs compatible with the layering.
    
3. **DAG sampled from the M-layering**
    
    Represented as families separated by `|`, with the first label being the child and the following (if any) the parents.
     
4. **Score of the DAG sampled from the M-layering**

5. **Proposed move** 
    
    Possible values are
    
    - `R_basic_move`
    - `R_swap_any`
    - `B_swap_nonadjacent`
    - `B_swap_adjacent`
    - `B_relocate_many`
    - `B_relocate_one`
    - `stay`
    
    `R_` and `B_` prefixes indicate moves in root-partition and layering space, respectfully. The `stay` value indicates that no move was proposed.
    
    The moves can further be prefixed by
    - `invalid_input_`
   
       to indicate that for the randomly selected move the initial M-layering is invalid, e.g., the number of layers is less than 3 for `B_swap_nonadjacent`,
       
    - `invalid_output_`
   
       to indicate that the proposed layering is not a valid M-layering (should be possible only with `B_relocate_many` as further explained in the source),
       
    - `identical_`
   
       to indicate that the proposed move in the root-partition space maps back to the initial M-layering.

6. **Whether the move was accepted (`1`) or not (`0`)**
7. **Acceptance probability for the proposed move**

	The probability is not truncated to max 1, so it can take any value between 0 and infinity. 
    
8. **Time in seconds used to evaluate the parent-sums function**
9. **Time in seconds used to compute the layering score**
    
    Excluding time used for evaluating the parent-sums function.
    
Items 6 to 9 are left blank, i.e., `None` if the proposed move was `stay` or if it was prefixed by any of `invalid_input_`, `invalid_output_` or `identical_`. 

### TODO

- Edge reversal move
- Effective moves only (no more `invalid_output_` or `identical_`)
- Time critical parts in C++
- More "production quality" implementation overall

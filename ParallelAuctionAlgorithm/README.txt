In this Project i parallelized the Auction algorithm for solving the linear Assignment Problem.

Describe the linear Assignment Problem:
In the LAP n resources have to be assigned to n tasks on a one-to-one basis. Let A(i) be
a nonempty set of tasks which can be assigned to a resource i. There is a fixed cost cij for
matching resource i with task j ∈ A(i). A set of k resource-task pairs S = (i1, j1), . . . , (ik, jk)
is called an assignment, if the following conditions are met: 0 ≤ k ≤ n, jm ∈ A(im) for each k,
and the resources i1, . . . , ik and tasks j1, . . . , jk are all distinct. A resource i is called assigned
if there exists a pair (i, j) ∈ S, otherwise resource i is unassigned. The objective of the LAP
is to minimize the total costs of the assignment, which is the sum Pk
m=1 cimjm of the cost of
the assigned pairs. A complete assignment contains k = n resource-task pairs, otherwise it is
incomplete (k < n). Under the assumption that there exists at least one complete assignment,
the goal is to find a complete assignment with minimum total cost. 

Auction algorithm: 
In 1988 Bertsekas proposed the auction algorithm in his article “The auction algorithm:
A distributed relaxation method for the assignment problem” [11] and is inspired by real
auction processes.

The decentralized nature of the
biding process makes the algorithm well-suited for parallel or distributed implementation.
This property allows that simultaneous bids by multiple resources are executed by multiple
processors.




Adjust thread number and problem sizes in according source.cpp files

Number of threads:
#define NUM_THREADS 8

Problem sizes:
vector<int> test_cases = {...}

execute files with the arguments:
./filename num_test_runs output_filename

Output files get stored in data Directory

First file stores all times of the test runs
Second file stores the median, and average runtimes for each used Problem size. 
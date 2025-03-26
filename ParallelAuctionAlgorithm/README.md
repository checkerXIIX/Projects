# Parallelized Auction Algorithm for the Linear Assignment Problem

## Project Overview
This project implements a **parallelized version** of the **Auction Algorithm** to efficiently solve the **Linear Assignment Problem (LAP)**. The parallelization leverages the decentralized nature of the bidding process, allowing multiple resources to bid simultaneously, optimizing performance on multi-threaded architectures.

---

## Linear Assignment Problem (LAP)
The **Linear Assignment Problem** consists of assigning **n resources** to **n tasks** on a **one-to-one basis**, minimizing the total assignment cost.

### Problem Definition
- Each resource \( i \) has a set of assignable tasks \( A(i) \).
- Assigning resource \( i \) to task \( j \) incurs a cost \( c_{ij} \).
- A valid assignment consists of **distinct resource-task pairs** \( (i, j) \).
- The objective is to find a **complete assignment** (one resource per task) that **minimizes the total cost**:

  \[
  \sum_{m=1}^{k} c_{i_m j_m}
  \]

- If \( k = n \), the assignment is **complete**; otherwise, it is **incomplete**.

---

## Auction Algorithm
The **Auction Algorithm**, proposed by Bertsekas in 1988, is inspired by real-world **auction processes**. It operates as follows:

1. Resources **bid** for tasks, increasing their bid price.
2. Tasks are assigned to the highest bidder.
3. The process iterates until a complete assignment is found.

### Why Parallelize?
The algorithm is inherently suited for **parallelization** because:
- **Decentralized bidding** allows independent computations.
- **Multiple resources** can place bids **simultaneously**, reducing execution time.
- **Multi-threaded implementation** significantly accelerates large-scale LAP instances.

---

## Usage

### Configuration
Modify the following parameters in the source files:

- **Number of Threads**
  ```cpp
  #define NUM_THREADS 8

- **Problem Sizes**
  ```cpp
  vector<int> test_cases = {...};

### Execution
Run the executable with the following arguments:
    ```bash
    ./filename num_test_runs output_filename

### Output Files
Results are stored in the `/data` directory:
    1. **First file** → Stores all individual test run times.
    2. **Second file** → Stores median and average runtimes for each problem size.

## Acknowledgments
This project is based on Bertsekas' work on the Auction Algorithm. For further details, refer to:
    - D. P. Bertsekas (1988). The Auction Algorithm: A Distributed Relaxation Method for the Assignment Problem.


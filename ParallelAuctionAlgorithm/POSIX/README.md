# POSIX Threads (PThreads) Parallel Auction Algorithm

## Overview
This implementation parallelizes the **Auction Algorithm** using **POSIX Threads (PThreads)** with an **asynchronous Jacobi approach**. This method allows **threads to operate independently**, reducing idle time and **maximizing resource utilization**.

---

## Parallelization Strategy

### 1. **Task Management with a Shared Stack**
- A **shared stack** manages the **indices of unassigned resources**.
- Before execution:
  - The stack is **cleared and repopulated** with unassigned resource indices.
- Threads dynamically **fetch tasks** from this stack during execution.

### 2. **Thread Execution (Asynchronous Jacobi Strategy)**
- Each thread runs in an **infinite loop** with the following logic:
  1. **Check if the stack is empty**:
     - If **empty**, the thread **exits** and synchronizes with others.
     - If **non-empty**, the thread **pops an index** from the stack.
  2. **Mutex Protection**:
     - The **stack operation** (checking & popping) is protected by a **mutex** to **prevent race conditions**.
  3. **Bid Calculation**:
     - The thread determines the **best task** for the resource and **calculates the bid amount**.
  4. **Assignment Phase**:
     - If the bid **exceeds the current task price**, the **resource is assigned** to the task.
     - The **assignment process is mutex-protected** to avoid race conditions.

---

## Key Features
- **Asynchronous Execution** → Threads operate independently for better performance.  
- **Dynamic Task Fetching** → Threads **self-assign** work, reducing idle time.  
- **Mutex Protection** → Ensures **safe access** to shared data structures.  

---

## Implementation Notes
- The number of threads is **configurable**.
- The **shared stack** dynamically holds unassigned resources.
- **Mutex synchronization** prevents race conditions in both **bidding and assignment** phases.

---

## References
- The Figure provides a detailed breakdown of the parallel logic.

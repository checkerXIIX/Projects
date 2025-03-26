# Serial Auction Algorithm for the Linear Assignment Problem

## Overview
The **Auction Algorithm** is an iterative approach to solving the **Linear Assignment Problem (LAP)** by incrementally assigning resources to tasks. Each iteration refines a **partial assignment** while maintaining a **price vector** that satisfies the **ϵ-Complementary Slackness (ϵ-CS) condition**.

---

## Algorithm Workflow
The algorithm consists of **two main phases**, repeated until a **complete assignment** is found:

### 1. **Bidding Phase**
- Select an **unassigned resource** \( i \).
- Find the **best task** \( j \) that provides the highest **net benefit**:
  \[
  \text{net benefit} = c_{ij} - p_j
  \]
- Find the **second-best task** \( k \) that provides the second-highest net benefit.
- Compute the **bid** of resource \( i \):
  \[
  b_{ij} = p_j + (c_{ij} - p_j) - (c_{ik} - p_k) + \epsilon
  \]

### 2. **Assignment Phase**
- Set the **price** of task \( j \) to the **highest bid**:
  \[
  p_j := \max b_{ij}
  \]
- If task \( j \) was **previously assigned** to another resource \( r \), remove that assignment and mark \( r \) as **unassigned**.
- Assign resource \( i_j \) (the highest bidder) to task \( j \).

This process repeats **until all resources are assigned**.

---

## Implementation Notes
- The algorithm **sequentially processes** one resource at a time, making it **suitable for single-threaded execution**.
- The choice of **ϵ** influences convergence speed and solution accuracy.
- A complete assignment is achieved when **every resource is assigned to a task** with a corresponding price.

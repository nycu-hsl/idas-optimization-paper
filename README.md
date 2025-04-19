ML-based Intrusion Detection as a Service: Traffic Split Offloading and Cost Allocation in a Multi-tier Architecture
===
This project is the repository of our paper, entitled ML-based Intrusion Detection as a Service: Traffic Split Offloading and Cost Allocation in a Multi-tier Architecture, which was published in IEEE Transactions on Service Computing. This study evaluates the performance of IDaS in a multi-tier architecture, utilizing traffic split offloading to enhance performance by mapping three in-sequence ML-based IDS tasks (pre-processing, binary detection, multi-class classification) to the architectures as the offloading destinations. We employ a simulated annealing-based traffic offloading and cost allocation (SA-TOCA) algorithm to determine the offloading ratio for each traffic path and the cost requirements for each tier.

## 🔧 Project Structure

The repository is organized into two main directories:

- **`cost_optimization/`** – Optimizes offloading ratios and cost allocation to **minimize computation cost**.
- **`delay_optimization/`** – Optimizes offloading ratios and cost allocation to **minimize average delay**.

Each directory contains:

- **4 optimization scripts**, one for each multi-tier architecture:
  - `fog_edge.py`
  - `fog_cloud.py`
  - `edge_cloud.py`
  - `fog_edge_cloud.py`

- **`parameter_config.py`** – Configuration file for all experiment parameters.
- **`random_generator.py`** – Implements random value generation under constraints, as described in the paper.

---

## ⚙️ How to Run the Code

1. **Configure Parameters**

  Before running any optimization script, open and configure parameter_config.py according to your desired setup. The parameter values should match those described in the manuscript (e.g., traffic rate, number of nodes, service capacity, delay thresholds, etc.).

2. **Run Optimization Script**

   Navigate to the corresponding directory and run the script for the architecture you want to test. For example:

   ```bash
   cd cost_optimization
   python edge_cloud.py
   
## 📘 Code Structure (Each Architecture Script)

Each optimization script (e.g., `edge_cloud.py`) consists of the following six components:

1. **Architecture Resource Mapping**  
   Defines the task-to-tier mapping based on **Table III** from the paper, indicating where each task (pre-processing, binary detection, multi-class classification) is processed.

2. **Input Variable Initialization**  
   Loads input parameters (e.g., arrival rate, task size, service capacity) from `parameter_config.py`.

3. **Delay Calculation Function**  
   Computes the total delay using the formula defined in **Section III.D** of the manuscript.

4. **Cost Allocation Optimization Function**  
   Optimizes how the computation cost is distributed across the selected tiers under the current offloading configuration.

5. **Offloading Ratio Optimization Function**  
   Adjusts the traffic distribution ratios across fog, edge, and cloud to minimize cost or delay.

6. **Main Function**  
   Coordinates the execution of the above components and performs the full optimization process.

## 📚 Citation

If you use our code or results in your research, please cite our paper as:

**Didik Sudyana, Yuan-Cheng Lai, Ying-Dar Lin, Piotr Chołda**,  
*"ML-based Intrusion Detection as a Service: Traffic Split Offloading and Cost Allocation in a Multi-tier Architecture,"*  
IEEE Transactions on Services Computing, to appear.

### BibTeX
```bibtex
@article{sudyana2024idas,
  author    = {Didik Sudyana and Yuan-Cheng Lai and Ying-Dar Lin and Piotr Chołda},
  title     = {ML-based Intrusion Detection as a Service: Traffic Split Offloading and Cost Allocation in a Multi-tier Architecture},
  journal   = {IEEE Transactions on Services Computing},
  year      = {2024},
  note      = {To appear}
}



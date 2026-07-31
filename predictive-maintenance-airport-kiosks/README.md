

# 🛠️ Predictive Maintenance & Operational Analytics for Airport Kiosks

An end-to-end Data Science & Machine Learning project designed to analyze event logs, filter operational noise, and predict hardware/system failures for airport machines (hosts).

## 📊 Business Context & Key Insights
* **Operational vs Failure Noise**: High occurrence of `"No Paper"` status was identified not as a hardware defect, but as an operational routine performed by agents at flight closure.
* **Temporal Patterns**: Event activity peaks sharply around **13:00 (1 PM)** (exceeding 1,400 events/hr).
* **Target Metric**: Real failures (`Offline`, `Power Off`, `Unreachable`, etc.) isolated for failure prediction.

## 🚀 ML Model Performance (Tuned Random Forest)
By engineering rolling features (`Error_Count_Last_3`, `Mean_Time_Delta_3`), the predictive maintenance model achieved significant performance gains:

| Metric | Baseline Model | Tuned Model + Rolling Features |
| :--- | :---: | :---: |
| **Accuracy** | 88% | **96%** |
| **Recall (Errors)** | 66% | **85%** |
| **Precision (Errors)** | 63% | **90%** |
| **F1-Score** | 0.65 | **0.87** |

* **Top Predictive Feature**: `Error_Count_Last_3` (~65% importance), proving that recent error bursts are strong early indicators of full system failures.

## 🛠️ Project Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/predictive-maintenance-airport-kiosks.git](https://github.com/your-username/predictive-maintenance-airport-kiosks.git)
   cd predictive-maintenance-airport-kiosks
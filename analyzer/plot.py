import json
import argparse
import matplotlib.pyplot as plt

# Load the JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Extract metrics from the JSON data
def extract_metrics(json_data):
    metrics = {"precision": [], "recall": [], "f1-score": [], "accuracy": []}
    for entry in json_data:
        model_metrics = entry[0]
        metrics["precision"].append((model_metrics["0"]["precision"] + model_metrics["1"]["precision"]) / 2)
        metrics["recall"].append((model_metrics["0"]["recall"] + model_metrics["1"]["recall"]) / 2)
        metrics["f1-score"].append((model_metrics["0"]["f1-score"] + model_metrics["1"]["f1-score"]) / 2)
        metrics["accuracy"].append(model_metrics["accuracy"])
    return metrics

# Plot metrics for all models
def plot_all_metrics(metrics):
    steps = list(range(1, len(metrics["precision"]) + 1))
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric_name, metric_values in metrics.items():
        plt.plot(steps, metric_values, label=metric_name, marker='o')

    # Graph customization
    plt.title("Model Comparison Across Metrics")
    plt.xlabel("Model Steps")
    plt.ylabel("Metric Values")
    plt.ylim(0.99, 1.00)  # Adjust Y-axis based on your metric range
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Plot metrics from a JSON result file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON result file.")
    args = parser.parse_args()

    # Load data
    file_path = args.json_file
    data = load_json(file_path)

    # Extract metrics
    metrics = extract_metrics(data)

    # Plot all metrics
    plot_all_metrics(metrics)

if __name__ == "__main__":
    main()

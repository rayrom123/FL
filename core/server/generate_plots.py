import os
import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    accuracies = []
    losses = []
    
    if not os.path.exists(log_path):
        return None, None
        
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Tìm các dòng Global Model Round X | Accuracy: A% | Loss: L
    pattern = r"Global Model Round \d+ \| Accuracy: ([\d.]+)% \| Loss: ([\d.]+)"
    matches = re.findall(pattern, content)
    
    for acc, loss in matches:
        accuracies.append(float(acc))
        losses.append(float(loss))
        
    return accuracies, losses

def save_plots(run_dir, accuracies, losses):
    if not accuracies:
        return
        
    rounds = range(1, len(accuracies) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracies, 'b-o', label='Global Accuracy')
    plt.title(f'Accuracy - {os.path.basename(run_dir)}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds, losses, 'r-o', label='Global Loss')
    plt.title(f'Loss - {os.path.basename(run_dir)}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Created plot: {plot_path}")

def main():
    results_dir = "d:/FL/results"
    if not os.path.exists(results_dir):
        print("Results directory not found.")
        return
        
    for folder in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, folder)
        if os.path.isdir(run_dir):
            # Thử tìm log.txt hoặc training result1.txt
            log_files = ["log.txt", "training_log.txt", "training result1.txt"]
            found = False
            for log_name in log_files:
                log_path = os.path.join(run_dir, log_name)
                if os.path.exists(log_path):
                    print(f"Processing {run_dir} using {log_name}...")
                    accs, losses = parse_log_file(log_path)
                    if accs:
                        save_plots(run_dir, accs, losses)
                        found = True
                        break
            if not found:
                print(f"No valid log file found in {run_dir}")

if __name__ == "__main__":
    main()

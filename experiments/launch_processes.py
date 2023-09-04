import subprocess
import multiprocessing
import time

# Parameters
num_gpus = 6
processes_per_gpu = 4

# Global counters
started_counter = multiprocessing.Value("i", 0)
completed_counter = multiprocessing.Value("i", 0)
error_counter = multiprocessing.Value("i", 0)

# Keep track of GPUs in use and their count of tasks
gpu_locks = [multiprocessing.Lock() for _ in range(num_gpus)]
gpu_task_counts = [multiprocessing.Value("i", 0) for _ in range(num_gpus)]


def get_next_available_gpu():
    while True:
        for idx in range(1, num_gpus):  # skip GPU 0
            if gpu_task_counts[idx].value < processes_per_gpu:
                with gpu_locks[idx]:
                    if gpu_task_counts[idx].value < processes_per_gpu:
                        gpu_task_counts[idx].value += 1
                        return idx
        time.sleep(1)  # Sleep for 1 second if no available GPU is found


def launch_process(task):
    global started_counter
    global completed_counter
    global error_counter

    (
        epochs,
        num_timeframes,
        theta,
        num_negative_samples,
        last_task_accuracy,
        out_filename,
        total_tasks,
    ) = task
    gpu_idx = get_next_available_gpu()

    command = f"python peagat_solver_bpr.py --gpu_idx={gpu_idx} --epochs={epochs} --num_timeframes={num_timeframes} --theta={theta} --num_negative_samples={num_negative_samples} --last_task_accuracy={last_task_accuracy} --out_filename={out_filename}"

    with started_counter.get_lock():
        started_counter.value += 1
        print(f"STARTED ({started_counter.value}/{total_tasks}): {command}")

    try:
        subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        with completed_counter.get_lock():
            completed_counter.value += 1
            print(f"COMPLETED ({completed_counter.value}/{total_tasks}): {command}")

    except subprocess.CalledProcessError as e:
        with error_counter.get_lock():
            error_counter.value += 1
            print(f"ERROR ({error_counter.value}/{total_tasks}) in Command: {command}")
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
    finally:
        with gpu_locks[gpu_idx]:
            gpu_task_counts[gpu_idx].value -= 1


def main():
    base_num_negative_samples = 8
    computational_budgets = [4, 8, 16, 32, 64]
    num_timeframes_list = [4, 8, 16, 32]
    theta_list = [1, 2, 4, 8]
    last_task_accuracies = ["true", "false"]

    all_tasks = []

    for computational_budget in computational_budgets:
        for last_task_accuracy in last_task_accuracies:
            for theta in theta_list:
                for num_timeframes in num_timeframes_list:
                    num_negative_samples = base_num_negative_samples // theta
                    epochs = computational_budget // theta

                    if num_negative_samples == 0 or epochs == 0:
                        continue

                    out_filename = f"budget_{computational_budget}_theta_{theta}_timeframes_{num_timeframes}_last_{last_task_accuracy}"
                    all_tasks.append(
                        (
                            epochs,
                            num_timeframes,
                            theta,
                            num_negative_samples,
                            last_task_accuracy,
                            out_filename,
                            len(all_tasks),
                        )
                    )

    print(f"Total tasks generated: {len(all_tasks)}")
    print("Initializing multiprocessing pool...")
    with multiprocessing.Pool(processes=num_gpus * processes_per_gpu) as pool:
        pool.map(launch_process, all_tasks)
    print("Multiprocessing pool finished.")


if __name__ == "__main__":
    main()

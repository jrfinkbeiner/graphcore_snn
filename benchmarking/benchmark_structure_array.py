import yappi
import tonic
import tonic.transforms as transforms


def generate_tonic_nmnist_dataset(save_to):
    # transform = transforms.Compose([transforms.ToFrame(time_window=1000.0)])
    # transform = transforms.Compose([transforms.ToFrame(tonic.datasets.NMNIST.sensor_size, n_time_bins=100)])
    # transform = transforms.Compose([transforms.ToFrame(tonic.datasets.NMNIST.sensor_size, time_window=1000.0)])
    transform = None
    dataset = tonic.datasets.NMNIST(save_to=save_to,
                                    train=False,
                                    transform=transform,
                                    )
    return dataset

def get_timing(dataset, num_iterations=1000):
    yappi.set_clock_type("cpu")
    yappi.clear_stats()
    yappi.start()
    for i in range(num_iterations):
        b = dataset[i]
    yappi.stop()
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()
    
if __name__ == "__main__":
    save_to = "/Data/pgi-15/datasets/"
    dataset = generate_tonic_nmnist_dataset(save_to)
    get_timing(dataset)
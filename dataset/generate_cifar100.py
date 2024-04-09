import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 100
dir_path = "Cifar100/"


# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


def generate_cifar100_with_drift(dir_path, num_clients, num_classes, niid, balance, partition, iterations=3):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    change_point_str = 'rand'
    drift_together = 1
    train_iteration = 10
    stretch_factor = 5
    num_client = 20

    # Randomly generate a single change point for each clientI
    if change_point_str == 'rand':
        if drift_together == 1:
            #cp = np.random.random_sample() * train_iteration
            cp = np.random.randint(1, train_iteration//stretch_factor)
            change_point_per_client = [cp for c in range(num_client)]
        else:
            change_point_per_client = [np.random.randint(1, train_iteration//stretch_factor)
                                       for c in range(num_client)]
        
        # matrix of the concept in the training data for each time, client.
        # restricted to concept changes at time step boundary
        change_point = np.zeros((train_iteration//stretch_factor + 1, num_client))
        for c in range(num_client):
            t = change_point_per_client[c]
            change_point[t:,c] = 1
        #np.savetxt("./../../../data/changepoints/rand.cp", change_point, fmt='%u')
        
    config_path = os.path.join(dir_path, "config.json")
    if check(config_path, dir_path, dir_path, num_clients, num_classes, niid, balance, partition):
        return
        
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=os.path.join(dir_path, "rawdata"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=os.path.join(dir_path, "rawdata"), train=False, download=True, transform=transform)

    # Loading all data into memory
    train_images, train_labels = next(iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)))
    test_images, test_labels = next(iter(torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)))


    mnist = MNIST_Data()
    for it in range(train_iteration + 1):
        for c in range(num_client):
            k = change_point[it//stretch_factor][c]
            train_data = mnist.generate_sample(sample_per_client_iter, k)
            add_noise(train_data, noise_prob)
            # Save the data as files
            pd.DataFrame(train_data).to_csv(
                data_path + 'client_{}_iter_{}.csv'.format(c, it),
                index = False)

    for iteration in range(iterations):
        print(f"Generating data for iteration {iteration}")
        # Implement the logic to vary class distribution here
        # For simplicity, let's just shuffle the data to simulate drift
        shuffled_indices = np.random.permutation(len(train_labels))
        train_images, train_labels = train_images[shuffled_indices], train_labels[shuffled_indices]

        # Here, implement the logic to allocate data to clients based on the current iteration
        # This example does not explicitly change the class distribution among clients,
        # but you should modify the data allocation logic to reflect the intended concept drift
        X, y, statistic = separate_data((train_images.numpy(), train_labels.numpy()), num_clients, num_classes, 
                                        niid, balance, partition, class_per_client=20)

        # Split and save the data for this iteration
        train_data, test_data = split_data(X, y)
        iteration_dir = os.path.join(dir_path, f"iteration_{iteration}")
        save_file(os.path.join(iteration_dir, "config.json"), os.path.join(iteration_dir, "train"), 
                  os.path.join(iteration_dir, "test"), train_data, test_data, num_clients, num_classes, 
                  statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition)
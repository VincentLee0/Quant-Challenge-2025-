import matplotlib.pyplot as plt

def plot(numbers1, numbers2, name):
    plt.figure(figsize=(8, 4))
    plt.grid(True)
    plt.title("Line Graph")
    plt.xlabel("Index")
    plt.ylabel("Return Rate")
    if not numbers2:
        plt.ylim(min(numbers1), max(numbers1))
        plt.plot(numbers1, marker='o', linestyle='-')
    else:
        plt.ylim(min(numbers1 + numbers2), max(numbers1 + numbers2))
        plt.plot(numbers1, marker='o', linestyle='-', label='real data')
        plt.plot(numbers2, marker='o', linestyle='-', label='predicted data')
        plt.legend()
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.close()

